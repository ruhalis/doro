import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import os
from PIL import Image
import numpy as np
from models.attgan import Generator, Discriminator
from dataset import CustomDataset, get_transforms
from tqdm import tqdm
import pandas as pd
import time
from models.vgg_perceptual_loss import VGGPerceptualLoss
import torch.nn.functional as F
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.metrics import mean_squared_error
import lpips
from torch.cuda.amp import GradScaler
import datetime
import json
from torch.optim.lr_scheduler import LambdaLR
import shutil
import traceback

class SharpenessLoss(nn.Module):
    def __init__(self):
        super(SharpenessLoss, self).__init__()
        self.laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
    def forward(self, x):
        # Convert to grayscale if input is RGB
        if x.shape[1] == 3:
            x = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
            x = x.unsqueeze(1)
        
        # Apply Laplacian filter
        laplacian_kernel = self.laplacian_kernel.to(x.device)
        laplacian_kernel = laplacian_kernel.repeat(1, x.shape[1], 1, 1)
        edge_map = F.conv2d(x, laplacian_kernel, padding=1)
        
        # Encourage stronger edges (negative loss because we want to maximize sharpness)
        return -torch.mean(torch.abs(edge_map))

class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()
        
    def forward(self, x):
        # Calculate gradients
        h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
        w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        return h_tv + w_tv

def download_and_verify_weights(config, gpu_id):
    """Download and verify model weights"""
    if gpu_id == 0:  # Only print on main process
        print("\nVerifying pre-trained model weights...")
        
    try:
        # Import torchvision here to avoid circular imports
        from torchvision import models
        
        # VGG weights for perceptual loss
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        if gpu_id == 0:
            print("✓ VGG16 weights loaded successfully")
        
        # Face recognition model
        face_model = InceptionResnetV1(pretrained='vggface2')
        if gpu_id == 0:
            print("✓ FaceNet weights loaded successfully")
        
        # MTCNN for face detection
        mtcnn = MTCNN(device=torch.device(f'cuda:{gpu_id}'))
        if gpu_id == 0:
            print("✓ MTCNN weights loaded successfully")
        
        # LPIPS
        loss_fn_alex = lpips.LPIPS(net='alex')
        if gpu_id == 0:
            print("✓ LPIPS weights loaded successfully")
        
        return True
        
    except Exception as e:
        if gpu_id == 0:  # Only print on main process
            print(f"❌ Error downloading weights: {str(e)}")
            print("\nTroubleshooting steps:")
            print("1. Check internet connection")
            print("2. Verify torch hub cache directory permissions")
            print("3. Try clearing the cache: torch.hub.clear_cache()")
        return False

class AttGANTrainer:
    def __init__(self, config, gpu_id):
        """Initialize trainer with proper use of all config parameters"""
        self.config = config
        self.gpu_id = gpu_id
        
        # Set up device
        self.device = torch.device(f'cuda:{gpu_id}')
        
        # Clear CUDA cache if configured
        if self.config['clear_cache_on_start']:
            torch.cuda.empty_cache()
        
        # Set custom torch hub directory
        if self.config['torch_hub_dir']:
            torch.hub.set_dir(self.config['torch_hub_dir'])
        
        # Initialize models and move to device
        self.initialize_models()
        
        # Setup data loaders with proper normalization
        self.setup_dataloaders()
        
        # Initialize losses and metrics
        self.initialize_losses()
        
        # Load pretrained weights
        self.load_pretrained_weights()
        
        # Calculate adaptive logging frequency
        self.calculate_logging_frequency()

    def initialize_models(self):
        """Initialize models with proper configuration"""
        self.G = Generator(
            input_channels=3,
            output_channels=3,
            base_filters=64
        ).to(self.device)
        
        self.D = Discriminator(
            input_channels=3,
            base_filters=64
        ).to(self.device)
        
        if self.config['world_size'] > 1:
            self.G = DDP(self.G, device_ids=[self.gpu_id])
            self.D = DDP(self.D, device_ids=[self.gpu_id])

    def setup_dataloaders(self):
        """Setup dataloaders with optimized loading parameters"""
        # Calculate optimal number of workers
        num_workers = min(
            os.cpu_count(),
            self.config['max_workers_per_gpu'] * torch.cuda.device_count()
        )
        
        transform_kwargs = {
            'image_size': self.config['image_size'],
            'normalize': self.config['normalize_samples'],
            'value_range': self.config['sample_value_range']
        }
        
        train_dataset = CustomDataset(
            self.config['dataset_path'],
            transform=get_transforms(**transform_kwargs, is_train=True)
        )
        
        val_dataset = CustomDataset(
            self.config['dataset_path'],
            transform=get_transforms(**transform_kwargs, is_train=False)
        )
        
        # Set up samplers for distributed training
        train_sampler = DistributedSampler(train_dataset) if self.config['world_size'] > 1 else None
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if self.config['world_size'] > 1 else None
        
        # Configure DataLoader with optimized settings
        dataloader_kwargs = {
            'batch_size': self.config['batch_size'],
            'num_workers': num_workers,
            'pin_memory': True,
            'prefetch_factor': self.config['prefetch_factor'],
            'persistent_workers': True,  # Keep workers alive between epochs
            'drop_last': True,  # Prevent issues with incomplete batches
        }
        
        if torch.cuda.is_available():
            # Enable non-blocking memory transfers
            dataloader_kwargs.update({
                'pin_memory_device': f'cuda:{self.gpu_id}',
                'pin_memory': True
            })
        
        self.train_loader = DataLoader(
            train_dataset,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            **dataloader_kwargs
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            sampler=val_sampler,
            **dataloader_kwargs
        )
        
        if self.gpu_id == 0:
            print(f"\nDataLoader Configuration:")
            print(f"  Number of workers: {num_workers}")
            print(f"  Prefetch factor: {self.config['prefetch_factor']}")
            print(f"  Batch size: {self.config['batch_size']}")
            print(f"  Training samples: {len(train_dataset)}")
            print(f"  Validation samples: {len(val_dataset)}")

    def normalize_batch(self, batch):
        """Normalize batch according to config settings"""
        if not self.config['normalize_samples']:
            return batch
        
        min_val, max_val = self.config['sample_value_range']
        return (batch - min_val) / (max_val - min_val) * 2 - 1

    def denormalize_batch(self, batch):
        """Denormalize batch according to config settings"""
        if not self.config['normalize_samples']:
            return batch
        
        min_val, max_val = self.config['sample_value_range']
        return (batch + 1) / 2 * (max_val - min_val) + min_val

    def initialize_losses(self):
        """Initialize losses with face recognition components"""
        try:
            # Initialize face detection and recognition
            self.mtcnn = MTCNN(
                image_size=self.config['image_size'],
                margin=0,
                device=self.device,
                keep_all=True
            ).to(self.device)
            
            self.face_model = InceptionResnetV1(
                pretrained='vggface2',
                device=self.device
            ).to(self.device)
            self.face_model.eval()
            
            # Other losses initialization
            self.perceptual_loss = VGGPerceptualLoss().to(self.device)
            self.sharpness_loss = SharpenessLoss().to(self.device)
            self.tv_loss = TotalVariationLoss().to(self.device)
            self.lpips_loss = lpips.LPIPS(net='alex').to(self.device)
            
            # Initialize face recognition loss weights
            self.identity_weight = self.config['lambda_identity']
            self.face_alignment_weight = self.config['lambda_face_alignment']
            
        except Exception as e:
            if self.gpu_id == 0:
                print(f"Error initializing face recognition: {str(e)}")
            raise

    def load_pretrained_weights(self):
        """Load pretrained weights with comprehensive error handling"""
        if self.gpu_id == 0:
            print("\nLoading pretrained weights...")
        
        try:
            # First, try loading from resume checkpoint if specified
            if self.config['resume_from_checkpoint']:
                checkpoint_path = os.path.join(
                    self.config['checkpoint_dir'],
                    f"checkpoint_epoch_{self.config['resume_from_checkpoint']}.pth"
                )
                if os.path.exists(checkpoint_path):
                    return self.load_checkpoint(checkpoint_path)
            
            # If no resume checkpoint, try loading pretrained weights
            pretrained_paths = [
                self.config.get('pretrained_weights_path', ''),  # User-specified path
                'pretrained/attgan_weights.pth',                 # Default path
                os.path.join(self.config['checkpoint_dir'], 'best_model.pth')  # Best model path
            ]
            
            for path in pretrained_paths:
                if path and os.path.exists(path):
                    if self.gpu_id == 0:
                        print(f"Loading pretrained weights from: {path}")
                    return self.load_checkpoint(path)
            
            # If no weights found, provide instructions
            if self.gpu_id == 0:
                print("\nNo pretrained weights found. To use pretrained weights:")
                print("1. Download weights from: https://github.com/your-repo/weights")
                print("2. Place them in the 'pretrained' directory")
                print("3. Or specify custom path in config['pretrained_weights_path']")
                print("\nStarting training from scratch...")
            
            return 0  # Start from epoch 0
            
        except Exception as e:
            if self.gpu_id == 0:
                print(f"\nError loading pretrained weights: {str(e)}")
                print("Starting from scratch...")
            return 0

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint with comprehensive error handling"""
        try:
            if self.gpu_id == 0:
                print(f"\nLoading checkpoint: {checkpoint_path}")
            
            # Try different loading strategies
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
            except RuntimeError as e:
                if "unexpected key" in str(e):
                    # Try loading with strict=False
                    checkpoint = torch.load(
                        checkpoint_path,
                        map_location=self.device,
                        weights_only=True
                    )
                else:
                    raise
            
            # Verify checkpoint contents
            required_keys = ['G_state_dict', 'D_state_dict']
            if not all(key in checkpoint for key in required_keys):
                raise ValueError(f"Checkpoint missing required keys: {required_keys}")
            
            # Load model states with version checking
            if isinstance(self.G, DDP):
                self.G.module.load_state_dict(checkpoint['G_state_dict'], strict=False)
                self.D.module.load_state_dict(checkpoint['D_state_dict'], strict=False)
            else:
                self.G.load_state_dict(checkpoint['G_state_dict'], strict=False)
                self.D.load_state_dict(checkpoint['D_state_dict'], strict=False)
            
            # Load optimizer states if available
            if all(key in checkpoint for key in ['g_optimizer', 'd_optimizer']):
                self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
                self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
            
            # Load scaler state if available
            if 'scaler' in checkpoint and hasattr(self, 'scaler'):
                self.scaler.load_state_dict(checkpoint['scaler'])
            
            # Get starting epoch
            start_epoch = checkpoint.get('epoch', 0) + 1
            
            if self.gpu_id == 0:
                print(f"✓ Successfully loaded checkpoint from epoch {start_epoch-1}")
            return start_epoch
            
        except Exception as e:
            if self.gpu_id == 0:
                print(f"\n❌ Error loading checkpoint: {str(e)}")
                print("Starting from scratch...")
            return 0

    def train(self):
        """Main training loop with specific error handling"""
        try:
            start_epoch = self.load_checkpoint(self.config['resume_from']) if self.config['resume_from'] else 0
            
            for epoch in range(start_epoch, self.config['epochs']):
                if self.gpu_id == 0:
                    print(f"\nEpoch {epoch}/{self.config['epochs']}")
                
                try:
                    # Training epoch
                    self.train_epoch(epoch)
                    
                    # Validation step
                    if epoch % self.config['val_frequency'] == 0:
                        should_stop = self.validate(epoch)
                        if should_stop:
                            if self.gpu_id == 0:
                                print("Early stopping criterion met")
                            break
                    
                    # Save checkpoint
                    if epoch % self.config['save_frequency'] == 0:
                        self.save_checkpoint(epoch)
                        
                except torch.cuda.OutOfMemoryError as e:
                    self.handle_oom_error(epoch, str(e))
                    if self.config['save_on_error']:
                        self.save_checkpoint(epoch, filename='oom_error.pth')
                    continue
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        self.handle_oom_error(epoch, str(e))
                        if self.config['save_on_error']:
                            self.save_checkpoint(epoch, filename='oom_error.pth')
                        continue
                    else:
                        raise  # Re-raise other RuntimeErrors
                        
                except KeyboardInterrupt:
                    if self.gpu_id == 0:
                        print("\nTraining interrupted by user")
                        self.save_checkpoint(epoch, filename='interrupt.pth')
                    return
                    
        except Exception as e:
            if self.gpu_id == 0:
                print(f"\nCritical training error: {str(e)}")
                self.log_critical_error(e)
                if 'epoch' in locals():
                    self.save_checkpoint(epoch, filename='critical_error.pth')
            raise  # Re-raise the exception for proper debugging

    def log_critical_error(self, error):
        """Log critical errors with detailed information"""
        if self.gpu_id == 0:
            error_info = {
                'timestamp': datetime.datetime.now().isoformat(),
                'error_type': type(error).__name__,
                'error_message': str(error),
                'device': str(self.device),
                'memory_allocated': f"{torch.cuda.memory_allocated(self.device)/1024**3:.2f}GB",
                'max_memory': f"{torch.cuda.max_memory_allocated(self.device)/1024**3:.2f}GB",
                'traceback': traceback.format_exc()
            }
            
            # Log to file
            error_log_path = os.path.join(self.config['log_dir'], 'critical_errors.json')
            with open(error_log_path, 'a') as f:
                json.dump(error_info, f)
                f.write('\n')
            
            # Log to tensorboard
            if hasattr(self, 'writer'):
                self.writer.add_text('critical_error', str(error_info))

    def handle_oom_error(self, epoch, error_msg):
        """Handle out of memory errors with recovery attempts"""
        if self.gpu_id == 0:
            print(f"\nOOM error in epoch {epoch}: {error_msg}")
            print("Attempting recovery...")
        
        # Clear memory
        torch.cuda.empty_cache()
        
        # Log memory stats
        if self.gpu_id == 0:
            memory_stats = {
                'allocated': torch.cuda.memory_allocated(self.device)/1024**3,
                'cached': torch.cuda.memory_reserved(self.device)/1024**3,
                'max_allocated': torch.cuda.max_memory_allocated(self.device)/1024**3
            }
            
            print("\nMemory Stats (GB):")
            for key, value in memory_stats.items():
                print(f"  {key}: {value:.2f}")
            
            if hasattr(self, 'writer'):
                for key, value in memory_stats.items():
                    self.writer.add_scalar(f'memory/{key}', value, self.global_step)

    def process_large_image(self, image):
        """Process large images with consistent device handling"""
        B, C, H, W = image.shape
        chunk_size = self.config['process_chunk_size']
        
        # Ensure input is on correct device
        image = image.to(self.device, non_blocking=True)
        output = torch.zeros_like(image, device=self.device)
        
        for h in range(0, H, chunk_size):
            for w in range(0, W, chunk_size):
                h_end = min(h + chunk_size, H)
                w_end = min(w + chunk_size, W)
                chunk = image[:, :, h:h_end, w:w_end]
                
                with torch.cuda.amp.autocast(enabled=self.config['use_amp']):
                    output[:, :, h:h_end, w:w_end] = self.G(chunk)
        
        return output

    def train_epoch(self, epoch):
        """Training epoch with controlled sample saving"""
        self.G.train()
        self.D.train()
        
        # Save samples only at specified intervals
        should_save_samples = (
            epoch % self.config['sample_frequency'] == 0 or
            epoch < self.config['initial_sample_epochs'] or
            epoch >= self.config['epochs'] - self.config['final_sample_epochs']
        )
        
        # Initialize progress bar
        if self.gpu_id == 0:
            pbar = tqdm(
                total=len(self.train_loader),
                desc=f'Epoch {epoch}/{self.config["epochs"]}',
                leave=True
            )
        
        # Training loop with sample management
        for batch_idx, (real_A, real_B) in enumerate(self.train_loader):
            try:
                # Training step
                g_loss, d_loss = self.training_step(real_A, real_B)
                
                # Save samples if needed
                if should_save_samples and batch_idx == 0:
                    self.save_samples(epoch, real_A, real_B)
                
                # Update progress
                if self.gpu_id == 0:
                    pbar.update(1)
                    if batch_idx % self.config['log_frequency'] == 0:
                        pbar.set_postfix({
                            'G_loss': f'{g_loss:.4f}',
                            'D_loss': f'{d_loss:.4f}'
                        })
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.handle_oom_error(batch_idx, e)
                    continue
                raise

    def training_step(self, real_A, real_B):
        """Training step with gradient clipping"""
        try:
            # Ensure inputs are on correct device
            real_A = real_A.to(self.device, non_blocking=True)
            real_B = real_B.to(self.device, non_blocking=True)
            
            # Reset gradients
            self.g_optimizer.zero_grad()
            self.d_optimizer.zero_grad()
            
            for acc_step in range(self.config['gradient_accumulation_steps']):
                with torch.cuda.amp.autocast(enabled=self.config['use_amp']):
                    # Generate fake images
                    fake_B = self.G(real_A)
                    
                    # Train Discriminator
                    d_loss = self.train_discriminator(real_B, fake_B.detach())
                    d_loss = d_loss / self.config['gradient_accumulation_steps']
                    
                # Discriminator backward pass
                self.scaler.scale(d_loss).backward()
                
                # Clip D gradients
                if self.config['gradient_clip_value'] > 0:
                    self.scaler.unscale_(self.d_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.D.parameters(),
                        self.config['gradient_clip_value']
                    )
                
                with torch.cuda.amp.autocast(enabled=self.config['use_amp']):
                    # Train Generator
                    fake_B = self.G(real_A)
                    g_loss = self.train_generator(real_A, real_B, fake_B)
                    g_loss = g_loss / self.config['gradient_accumulation_steps']
                
                # Generator backward pass
                self.scaler.scale(g_loss).backward()
                
                # Clip G gradients
                if self.config['gradient_clip_value'] > 0:
                    self.scaler.unscale_(self.g_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.G.parameters(),
                        self.config['gradient_clip_value']
                    )
                
                # Memory management
                del fake_B
                torch.cuda.empty_cache()
            
            # Update weights with clipped gradients
            self.scaler.step(self.d_optimizer)
            self.scaler.step(self.g_optimizer)
            self.scaler.update()
            
            return g_loss.item(), d_loss.item()
            
        except Exception as e:
            self.handle_training_error(e)
            raise

    def should_clear_cache(self, batch_idx):
        """Memory management with device-specific tracking"""
        if batch_idx % self.config['empty_cache_frequency'] == 0:
            current_memory = torch.cuda.memory_allocated(self.device) / 1024**3
            max_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            if current_memory > self.config['memory_threshold'] * max_memory:
                return True
        return False

    def handle_oom_error(self, batch_idx, error):
        """Device-specific OOM handling"""
        print(f"\nWARNING: Out of memory on device {self.device} (batch {batch_idx})")
        print(f"Current memory: {torch.cuda.memory_allocated(self.device)/1024**3:.2f}GB")
        print(f"Max memory: {torch.cuda.max_memory_allocated(self.device)/1024**3:.2f}GB")
        
        torch.cuda.empty_cache()
        if hasattr(self, 'writer'):
            self.writer.add_scalar('training/oom_events', 1, self.global_step)

    def handle_training_error(self, batch_idx, error):
        """Handle other training errors"""
        print(f"\nERROR on batch {batch_idx}: {str(error)}")
        print(f"Memory usage at error: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        if hasattr(self, 'writer'):
            self.writer.add_scalar('training/error_events', 1, self.global_step)

    def log_progress(self, epoch, batch_idx, g_loss, d_loss):
        """Enhanced progress logging with gradient monitoring"""
        if self.gpu_id != 0:
            return
        
        if batch_idx % self.config['log_frequency'] == 0:
            step = epoch * len(self.train_loader) + batch_idx
            
            # Calculate gradient norms
            g_grad_norm = 0.0
            d_grad_norm = 0.0
            
            for p in self.G.parameters():
                if p.grad is not None:
                    g_grad_norm += p.grad.data.norm(2).item() ** 2
            g_grad_norm = g_grad_norm ** 0.5
            
            for p in self.D.parameters():
                if p.grad is not None:
                    d_grad_norm += p.grad.data.norm(2).item() ** 2
            d_grad_norm = d_grad_norm ** 0.5
            
            # Log metrics
            metrics = {
                'Train/G_loss': g_loss,
                'Train/D_loss': d_loss,
                'Train/G_grad_norm': g_grad_norm,
                'Train/D_grad_norm': d_grad_norm,
                'Train/G_grad_clipped': g_grad_norm > self.config['gradient_clip_value'],
                'Train/D_grad_clipped': d_grad_norm > self.config['gradient_clip_value']
            }
            
            # Log to tensorboard
            if hasattr(self, 'writer'):
                for name, value in metrics.items():
                    self.writer.add_scalar(name, value, step)

    def validate(self, epoch):
        """Validation with consistent device handling"""
        self.G.eval()
        metrics = {key: 0.0 for key in self.config['validation']['metric_weights']}
        
        try:
            with torch.no_grad():
                for batch_idx, (real_A, real_B) in enumerate(self.val_loader):
                    # Move data to device efficiently
                    real_A = real_A.to(self.device, non_blocking=True)
                    real_B = real_B.to(self.device, non_blocking=True)
                    
                    # Generate fake images
                    fake_B = self.process_large_image(real_A)
                    
                    # Compute metrics on device
                    metrics = self.compute_validation_metrics(
                        real_A, real_B, fake_B, metrics
                    )
                    
                    # Save validation images from main process
                    if batch_idx == 0 and self.gpu_id == 0:
                        self.save_validation_images(epoch, real_A, fake_B, real_B)
                    
        except Exception as e:
            self.handle_validation_error(e)
            return False
        
        return self.process_validation_results(epoch, metrics)

    def early_stopping_check(self, epoch, metrics):
        """Enhanced early stopping with multiple metrics"""
        improved = False
        
        # Define weights for different metrics in early stopping decision
        metric_weights = {
            'val_g_loss': 0.3,
            'val_fid': 0.3,
            'val_lpips': 0.2,
            'val_identity_sim': 0.2
        }
        
        # Calculate weighted score
        current_score = (
            metric_weights['val_g_loss'] * metrics['val_g_loss'] +
            metric_weights['val_fid'] * metrics['val_fid'] +
            metric_weights['val_lpips'] * metrics['val_lpips'] -  # Note: negative because higher is better
            metric_weights['val_identity_sim'] * metrics['val_identity_sim']  # Note: negative because higher is better
        )
        
        if not hasattr(self, 'best_score'):
            self.best_score = float('inf')
        
        if current_score < self.best_score:
            self.best_score = current_score
            improved = True
            self.patience_counter = 0
            
            # Save best model
            if self.gpu_id == 0:
                self.save_checkpoint(epoch, is_best=True)
                print(f"\nNew best model saved!")
                print(f"Metrics:")
                for key, value in metrics.items():
                    print(f"  {key}: {value:.4f}")
        else:
            self.patience_counter += 1
            if self.gpu_id == 0:
                print(f"\nNo improvement for {self.patience_counter} epochs")
                print(f"Current score: {current_score:.4f}")
                print(f"Best score: {self.best_score:.4f}")
        
        # Check early stopping condition
        if self.patience_counter >= self.config['early_stopping_patience']:
            if self.gpu_id == 0:
                print(f"\nEarly stopping triggered after {epoch} epochs!")
                print(f"Best validation score: {self.best_score:.4f}")
            return True
        
        return False

    def log_validation_metrics(self, epoch, metrics):
        """Enhanced validation metrics logging"""
        stats = {
            'epoch': epoch,
            'timestamp': datetime.datetime.now().isoformat(),
            **metrics,
            'best_score': getattr(self, 'best_score', float('inf')),
            'patience_counter': self.patience_counter
        }
        
        # Save to CSV
        df = pd.DataFrame([stats])
        log_file = os.path.join(self.config['log_dir'], 'validation_metrics.csv')
        df.to_csv(log_file, mode='a', header=not os.path.exists(log_file))
        
        # Print current status
        print(f"\nValidation Metrics (Epoch {epoch}):")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint with proper error handling and cleanup"""
        try:
            checkpoint_dir = self.config['checkpoint_dir']
            checkpoint_path = os.path.join(
                checkpoint_dir, 
                f'checkpoint_epoch_{epoch}.pth'
            )
            
            # Prepare checkpoint data
            checkpoint = {
                'epoch': epoch,
                'G_state_dict': self.G.module.state_dict() if isinstance(self.G, DDP) else self.G.state_dict(),
                'D_state_dict': self.D.module.state_dict() if isinstance(self.D, DDP) else self.D.state_dict(),
                'g_optimizer': self.g_optimizer.state_dict(),
                'd_optimizer': self.d_optimizer.state_dict(),
                'config': self.config,
                'scaler': self.scaler.state_dict(),
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            # Save checkpoint with safe atomic write
            self.safe_save_checkpoint(checkpoint_path, checkpoint)
            
            # Save best model if needed
            if is_best:
                best_path = os.path.join(checkpoint_dir, 'best_model.pth')
                self.safe_save_checkpoint(best_path, checkpoint)
                if self.gpu_id == 0:
                    print(f"✓ Saved new best model")
            
            # Save latest for easy resume
            latest_path = os.path.join(checkpoint_dir, 'latest.pth')
            self.safe_save_checkpoint(latest_path, checkpoint)
            
            # Cleanup old checkpoints
            self.cleanup_old_checkpoints(self.config['keep_last_checkpoints'])
            
            # Log checkpoint info
            if self.gpu_id == 0:
                print(f"✓ Checkpoint saved: {checkpoint_path}")
                self.log_checkpoint_info(epoch, checkpoint_path, is_best)
            
        except Exception as e:
            if self.gpu_id == 0:
                print(f"❌ Error saving checkpoint: {str(e)}")
                self.log_error("checkpoint_save_error", str(e))

    def safe_save_checkpoint(self, path, checkpoint):
        """Save checkpoint with atomic write operation"""
        temp_path = path + '.tmp'
        try:
            # Save to temporary file first
            torch.save(checkpoint, temp_path)
            # Atomic rename operation
            os.replace(temp_path, path)
        except Exception as e:
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

    def cleanup_old_checkpoints(self, keep_last):
        """Enhanced checkpoint cleanup with better error handling"""
        if self.gpu_id != 0:  # Only run on main process
            return
        
        try:
            checkpoint_dir = self.config['checkpoint_dir']
            if not os.path.exists(checkpoint_dir):
                return
            
            # Get list of regular checkpoints (excluding best and latest)
            checkpoints = sorted([
                f for f in os.listdir(checkpoint_dir)
                if f.startswith('checkpoint_epoch_') and f.endswith('.pth')
            ], key=lambda x: int(x.split('_')[2].split('.')[0]))
            
            # Keep the specified number of most recent checkpoints
            if len(checkpoints) > keep_last:
                for checkpoint in checkpoints[:-keep_last]:
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
                    try:
                        if os.path.exists(checkpoint_path):
                            os.remove(checkpoint_path)
                            if self.config['detailed_logging']['system_metrics']:
                                print(f"Removed old checkpoint: {checkpoint}")
                    except OSError as e:
                        print(f"Warning: Could not remove checkpoint {checkpoint}: {e}")
                        self.log_error("checkpoint_cleanup_error", str(e))
            
            # Log cleanup results
            if hasattr(self, 'writer'):
                self.writer.add_scalar(
                    'system/checkpoints_kept',
                    len(checkpoints[-keep_last:]),
                    self.global_step
                )
                
        except Exception as e:
            print(f"Warning: Error during checkpoint cleanup: {e}")
            self.log_error("checkpoint_cleanup_error", str(e))

    def log_checkpoint_info(self, epoch, path, is_best):
        """Log checkpoint information to CSV and tensorboard"""
        if not hasattr(self, 'checkpoint_log'):
            self.checkpoint_log = []
        
        info = {
            'epoch': epoch,
            'path': path,
            'is_best': is_best,
            'timestamp': datetime.datetime.now().isoformat(),
            'size_mb': os.path.getsize(path) / (1024 * 1024)
        }
        
        self.checkpoint_log.append(info)
        
        # Save to CSV
        if self.config['log_to_file']:
            df = pd.DataFrame([info])
            log_path = os.path.join(self.config['log_dir'], 'checkpoint_log.csv')
            df.to_csv(log_path, mode='a', header=not os.path.exists(log_path))
        
        # Log to tensorboard
        if hasattr(self, 'writer'):
            self.writer.add_scalar('system/checkpoint_size_mb', info['size_mb'], epoch)

    def save_validation_images(self, epoch, real_A, fake_B, real_B):
        """Save sample validation images"""
        # Denormalize images from [-1, 1] to [0, 1]
        real_A = (real_A + 1.0) * 0.5
        fake_B = (fake_B + 1.0) * 0.5
        real_B = (real_B + 1.0) * 0.5
        
        # Make a grid of images
        img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
        
        # Create filename
        save_path = os.path.join(
            self.config['sample_dir'], 
            f'epoch_{epoch}.png'
        )
        
        # Save image grid
        save_image(
            img_sample, 
            save_path,
            nrow=self.config['batch_size'],
            normalize=False,  # Already normalized
            value_range=(0, 1)
        )
        
        # Log images to tensorboard if enabled
        if hasattr(self, 'writer'):
            self.writer.add_image(
                'Validation/real_A_fake_B_real_B', 
                img_sample[0],
                epoch,
                dataformats='CHW'
            )

    def log_to_csv(self, epoch, g_loss, d_loss):
        stats = {
            'epoch': epoch,
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'timestamp': time.time()
        }
        
        df = pd.DataFrame([stats])
        df.to_csv('training_log.csv', mode='a', header=not os.path.exists('training_log.csv'))

    def denormalize(self, x):
        """Denormalize from [-1, 1] to [0, 1]"""
        return (x + 1) / 2

    def compute_fid_score(self, real_images, fake_images):
        """Compute FID score between real and generated images"""
        if not self.use_fid:
            return 0.0  # Return dummy value if FID calculation is disabled
        
        self.fid.update(real_images, real=True)
        self.fid.update(fake_images, real=False)
        fid_score = self.fid.compute()
        self.fid.reset()
        return fid_score

    def compute_identity_similarity(self, real_images, fake_images):
        """Compute cosine similarity between face embeddings"""
        with torch.no_grad():
            # Get face embeddings
            real_embeddings = self.face_model(real_images)
            fake_embeddings = self.face_model(fake_images)
            
            # Normalize embeddings
            real_embeddings = F.normalize(real_embeddings, p=2, dim=1)
            fake_embeddings = F.normalize(fake_embeddings, p=2, dim=1)
            
            # Compute cosine similarity
            similarity = F.cosine_similarity(real_embeddings, fake_embeddings)
            
            return similarity.mean().item()

    def compute_lpips_distance(self, real_images, fake_images):
        """Compute LPIPS perceptual distance"""
        with torch.no_grad():
            distance = self.lpips_loss(real_images, fake_images)
            return distance.mean().item()

    def train_discriminator(self, real_B, fake_B):
        """Train discriminator"""
        # Real images
        real_validity = self.D(real_B)
        real_labels = torch.ones_like(real_validity).to(self.device)
        d_real_loss = self.adversarial_loss(real_validity, real_labels)
        
        # Fake images - already detached from generator
        fake_validity = self.D(fake_B)
        fake_labels = torch.zeros_like(fake_validity).to(self.device)
        d_fake_loss = self.adversarial_loss(fake_validity, fake_labels)
        
        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        return d_loss

    def train_generator(self, real_A, real_B, fake_B):
        """Train generator with enhanced identity preservation"""
        # Basic reconstruction losses
        g_rec_loss = self.reconstruction_loss(fake_B, real_B)
        g_perceptual_loss = self.perceptual_loss(fake_B, real_B)
        
        # Face recognition losses
        face_identity_loss = self.compute_face_identity_loss(real_B, fake_B)
        face_alignment_loss = self.compute_face_alignment_loss(real_B, fake_B)
        
        # Adversarial and texture losses
        fake_validity = self.D(fake_B)
        real_labels = torch.ones_like(fake_validity).to(self.device)
        g_adv_loss = self.adversarial_loss(fake_validity, real_labels)
        g_sharpness_loss = self.sharpness_loss(fake_B)
        g_tv_loss = self.tv_loss(fake_B)
        
        # Combined loss with face recognition components
        g_loss = (
            self.config['lambda_adv'] * g_adv_loss +
            self.config['lambda_rec'] * g_rec_loss +
            self.config['lambda_perceptual'] * g_perceptual_loss +
            self.config['lambda_identity'] * face_identity_loss +
            self.config['lambda_face_alignment'] * face_alignment_loss +
            self.config['lambda_sharpness'] * g_sharpness_loss +
            self.config['lambda_tv'] * g_tv_loss
        )
        
        return g_loss

    def compute_face_identity_loss(self, real_images, fake_images):
        """Compute identity preservation loss using face recognition"""
        with torch.no_grad():
            # Get face embeddings
            real_embeddings = self.face_model(real_images)
            fake_embeddings = self.face_model(fake_images)
            
            # Normalize embeddings
            real_embeddings = F.normalize(real_embeddings, p=2, dim=1)
            fake_embeddings = F.normalize(fake_embeddings, p=2, dim=1)
            
            # Identity loss is negative cosine similarity
            identity_loss = -F.cosine_similarity(
                real_embeddings, 
                fake_embeddings, 
                dim=1
            ).mean()
            
            return identity_loss

    def compute_face_alignment_loss(self, real_images, fake_images):
        """Compute face alignment loss using MTCNN landmarks"""
        try:
            # Detect facial landmarks
            real_landmarks = self.mtcnn.detect_landmarks(real_images)
            fake_landmarks = self.mtcnn.detect_landmarks(fake_images)
            
            if real_landmarks is None or fake_landmarks is None:
                return torch.tensor(0.0).to(self.device)
            
            # Convert landmarks to tensors
            real_landmarks = torch.tensor(real_landmarks).to(self.device)
            fake_landmarks = torch.tensor(fake_landmarks).to(self.device)
            
            # Compute MSE between landmark positions
            alignment_loss = F.mse_loss(real_landmarks, fake_landmarks)
            
            return alignment_loss
            
        except Exception as e:
            if self.gpu_id == 0:
                print(f"Warning: Face alignment computation failed: {str(e)}")
            return torch.tensor(0.0).to(self.device)

    def calculate_logging_frequency(self):
        """Calculate optimal logging frequency based on dataset size"""
        dataset_size = len(self.train_loader.dataset)
        batches_per_epoch = dataset_size // (self.config['batch_size'] * self.config['world_size'])
        
        # Aim for roughly 10-20 log entries per epoch
        suggested_frequency = max(1, batches_per_epoch // 15)
        
        # Update config with calculated frequency
        if self.gpu_id == 0:
            print(f"\nDataset size: {dataset_size}")
            print(f"Batches per epoch: {batches_per_epoch}")
            print(f"Suggested logging frequency: {suggested_frequency}")
            
        self.config['log_frequency'] = suggested_frequency

    def save_samples(self, epoch, real_A, real_B, max_samples=8):
        """Save sample images with space management"""
        if self.gpu_id != 0:  # Only save on main process
            return
        
        try:
            sample_dir = self.config['sample_dir']
            
            # Create epoch-specific directory
            epoch_dir = os.path.join(sample_dir, f'epoch_{epoch}')
            os.makedirs(epoch_dir, exist_ok=True)
            
            # Generate fake images
            with torch.no_grad():
                fake_B = self.G(real_A[:max_samples])
            
            # Save comparison grid
            comparison = self.create_comparison_grid(
                real_A[:max_samples],
                fake_B,
                real_B[:max_samples]
            )
            
            # Save with compression
            save_image(
                comparison,
                os.path.join(epoch_dir, f'comparison.jpg'),
                nrow=max_samples,
                normalize=True,
                quality=90  # JPEG quality
            )
            
            # Cleanup old sample directories if needed
            self.cleanup_old_samples()
            
        except Exception as e:
            print(f"Error saving samples: {str(e)}")
            self.log_error("sample_save_error", str(e))

    def cleanup_old_samples(self):
        """Manage sample directory size"""
        try:
            sample_dir = self.config['sample_dir']
            if not os.path.exists(sample_dir):
                return
            
            # Get list of epoch directories
            epoch_dirs = sorted([
                d for d in os.listdir(sample_dir)
                if d.startswith('epoch_')
            ], key=lambda x: int(x.split('_')[1]))
            
            # Keep only specified number of sample directories
            if len(epoch_dirs) > self.config['keep_last_samples']:
                # Always keep first and last epoch samples
                dirs_to_remove = epoch_dirs[1:-1][:-self.config['keep_last_samples']]
                
                for dir_name in dirs_to_remove:
                    dir_path = os.path.join(sample_dir, dir_name)
                    try:
                        shutil.rmtree(dir_path)
                    except OSError as e:
                        print(f"Warning: Could not remove sample directory {dir_name}: {e}")
                    
        except Exception as e:
            print(f"Warning: Error during sample cleanup: {e}")

    def monitor_dataloader_performance(self):
        """Monitor and log DataLoader performance metrics"""
        if not self.config['monitor_loading']['track_loading_time']:
            return
        
        # Measure loading time for one epoch
        start_time = time.time()
        data_loading_times = []
        
        for batch_idx, (real_A, real_B) in enumerate(self.train_loader):
            batch_time = time.time() - start_time
            data_loading_times.append(batch_time)
            start_time = time.time()
            
            # Log statistics periodically
            if batch_idx % self.config['monitor_loading']['loading_stats_frequency'] == 0:
                avg_time = np.mean(data_loading_times)
                if self.gpu_id == 0:
                    print(f"\nDataLoader Performance:")
                    print(f"  Average batch loading time: {avg_time:.3f}s")
                    print(f"  Batches per second: {1.0/avg_time:.2f}")
                    
                    if hasattr(self, 'writer'):
                        self.writer.add_scalar(
                            'performance/data_loading_time',
                            avg_time,
                            self.global_step
                        )
                    
                    # Warning for slow loading
                    if (self.config['monitor_loading']['warn_slow_loading'] and 
                        avg_time > 0.1 * self.config['batch_size']):
                        print("\nWarning: Data loading might be a bottleneck")
                        print("Consider increasing num_workers or prefetch_factor")

def setup(rank, world_size, config):
        # Set environment variables for master address and port
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12356'

    # Initialize process group without init_method
    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank
    )
    
    # Set device for this process
    torch.cuda.set_device(rank)

def get_linear_warmup_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        return 1.0
    return LambdaLR(optimizer, lr_lambda)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, config):
    try:
        # Update config with process-specific info
        config['rank'] = rank
        config['world_size'] = world_size
        
        # Set up error handling for CUDA devices
        torch.cuda.set_device(rank)
        
        setup(rank, world_size, config)
        
        # Create trainer instance
        trainer = AttGANTrainer(config, rank)
        
        # Train model
        trainer.train()
        
    except Exception as e:
        print(f"Error in rank {rank}: {str(e)}")
        raise e
    finally:
        if dist.is_initialized():
            cleanup()

if __name__ == "__main__":
    print("Starting training script...")
    # Configuration
    config = {
        'dataset_path': 'datasets/dataset',
        'checkpoint_dir': 'checkpoints',
        'sample_dir': 'samples',
        'log_dir': 'logs',
        
        # Training parameters - Adjusted for high resolution
        'batch_size': 1,                     # Reduced batch size
        'num_workers': 1,                    # Minimal workers
        'lr': 0.00001,            # Keep low learning rate for stability
        'beta1': 0.5,
        'beta2': 0.999,
        'epochs': 200,
        
        # Model parameters - Tuned for high resolution
        'image_size': 1024,        # Change from 1024 to 512
        'lambda_adv': 1.0,        # Reduced adversarial weight
        'lambda_rec': 10.0,         # High reconstruction weight
        'lambda_perceptual': 5.0,  # Strong perceptual loss
        'lambda_identity': 2.0,    # Strong identity preservation
        'lambda_sharpness': 0.5,    # Moderate sharpness
        'lambda_tv': 0.1,          # Low TV loss
        
        # Memory optimization - Critical for 1024x1024
        'use_amp': True,           # Keep AMP enabled
        'gradient_accumulation_steps': 16,    # Increased accumulation
        'batch_chunk_size': 256,             # Process in smaller chunks
        'process_chunk_size': 512,           # Chunk size for processing
        
        # Training schedule
        'val_frequency': 1,
        'save_frequency': 2,       # More frequent saving
        'log_frequency': None,       # Reduced logging frequency
        
        # Distributed training
        'dist_url': 'tcp://127.0.0.1:12356',
        'dist_backend': 'nccl',
        'multiprocessing_distributed': True,
        'pin_memory': True,
        'prefetch_factor': 1,                # Minimal prefetch
        
        # Learning rate scheduling
        'use_lr_scheduler': True,
        'lr_scheduler_patience': 8,
        'lr_scheduler_factor': 0.5,
        
        # Checkpoint handling
        'resume_from_checkpoint': None,
        'keep_last_checkpoints': 3,  # Reduced due to larger file sizes
        'save_best_model': True,
        
        # Training stability
        'gradient_clip_value': 0.5,      # Maximum gradient norm
        'gradient_clip_algorithm': 'norm',  # Use norm clipping
        'monitor_gradients': True,         # Track gradient statistics
        'clip_discriminator': True,        # Apply clipping to discriminator
        'clip_generator': True,            # Apply clipping to generator
        
        # Early stopping
        'early_stopping_patience': 15,
        'resume_training': True,
        
        # Memory cleanup
        'empty_cache_frequency': 50,      # Check memory less frequently
        'warmup_epochs': 3,
        'warmup_lr_factor': 0.1,
        'torch_hub_dir': 'pretrained_weights',  # Custom directory for downloaded weights
        'clear_cache_on_start': True,          # Whether to clear existing cache
        
        # Added parameters for image quality
        'normalize_samples': True,     # Whether to normalize images to [-1, 1]
        'sample_value_range': (0, 1),  # Input image value range
        # Memory management
        'memory_threshold': 0.8,       # Trigger cleanup at 80% memory usage
        'max_oom_retries': 3,         # Maximum OOM retries per batch
        
        'validation': {
            'fid_frequency': 10,      # How often to compute FID score
            'metric_weights': {       # Weights for early stopping decision
                'val_g_loss': 0.3,
                'val_fid': 0.3,
                'val_lpips': 0.2,
                'val_identity_sim': 0.2
            }
        },
        # Pretrained weights configuration
        'pretrained_weights_path': '',  # Custom path to pretrained weights
        'download_weights': True,       # Whether to download weights automatically
        'verify_weights': True,         # Whether to verify downloaded weights
        'weights_download_retry': 3,    # Number of download attempts
        
        # Logging and monitoring
        'log_frequency': None,  # Will be calculated automatically
        'min_logging_interval': 30,  # Minimum seconds between logs
        'detailed_logging': {
            'system_metrics': True,     # Log system metrics (GPU memory, etc.)
            'training_metrics': True,   # Log detailed training metrics
            'batch_metrics': False      # Log per-batch metrics (can be verbose)
        },
        
        # Progress display
        'use_tqdm': True,              # Use progress bars
        'log_to_file': True,           # Save logs to file
        'log_level': 'INFO',           # Logging detail level
        
        # Face recognition parameters
        'lambda_identity': 2.0,          # Weight for face identity loss
        'lambda_face_alignment': 1.0,    # Weight for facial landmark alignment
        'face_recognition': {
            'min_detection_confidence': 0.9,
            'use_landmarks': True,
            'identity_threshold': 0.7,    # Minimum identity similarity threshold
            'alignment_threshold': 0.1    # Maximum allowed landmark deviation
        },
        # Checkpoint management
        'checkpoint_dir': 'checkpoints',
        'keep_last_checkpoints': 3,
        'save_best_model': True,
        'checkpoint_frequency': 5,  # Save every N epochs
        'save_on_error': True,     # Save checkpoint when error occurs
        'compress_checkpoints': False,  # Option to compress checkpoints
        'backup_checkpoints': False,    # Option to backup to external storage
        # Sample management
        'sample_frequency': 5,           # Save samples every N epochs
        'initial_sample_epochs': 5,      # Always save samples for first N epochs
        'final_sample_epochs': 3,        # Always save samples for last N epochs
        'keep_last_samples': 10,         # Number of sample directories to keep
        'max_samples_per_class': 8,      # Maximum number of samples to save
        'sample_image_quality': 90,      # JPEG quality for saved samples
        'sample_management': {
            'compress_samples': True,    # Use JPEG compression
            'max_disk_usage_gb': 10,     # Maximum disk space for samples
            'cleanup_threshold': 0.9,    # Cleanup when disk usage exceeds 90%
            'keep_first_epoch': True,    # Always keep first epoch samples
            'keep_best_samples': True,   # Keep samples from best models
        },
        # Error handling
        'error_handling': {
            'save_on_error': True,          # Save checkpoint on error
            'max_oom_retries': 3,           # Maximum OOM retry attempts
            'log_critical_errors': True,     # Log critical errors
            'error_log_path': 'logs/errors',  # Path for error logs
            'recovery_batch_size': 1,        # Fallback batch size for OOM
            'memory_monitoring': True        # Monitor memory usage
        },
        # Debug settings
        'debug': {
            'trace_memory': False,          # Enable memory tracing
            'profile_execution': False,      # Enable profiling
            'verbose_errors': True,         # Detailed error messages
            'save_memory_stats': True       # Save memory statistics
        },
        # DataLoader optimization
        'max_workers_per_gpu': 4,        # Maximum workers per GPU
        'num_workers': None,             # Will be calculated automatically
        'prefetch_factor': 4,            # Number of batches to prefetch
        'dataloader': {
            'persistent_workers': True,   # Keep workers alive between epochs
            'pin_memory': True,          # Use pinned memory for faster transfer
            'non_blocking': True,        # Enable non-blocking transfers
            'timeout': 120,              # Increase timeout for large images
            'drop_last': True,           # Skip incomplete batches
            'worker_init_fn': None,      # Custom worker initialization
        },
        
        # Memory management for data loading
        'memory_pinning': {
            'pin_memory': True,
            'pin_memory_device': None,   # Set automatically based on GPU
            'pin_memory_max_bytes': None # Automatic based on available memory
        },
        
        # Performance monitoring
        'monitor_loading': {
            'track_loading_time': True,  # Monitor data loading time
            'loading_stats_frequency': 50,  # How often to log loading stats
            'warn_slow_loading': True,   # Warning for slow data loading
        }
    }

    # Create necessary directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['sample_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)

    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    # Get number of available GPUs
    world_size = torch.cuda.device_count()
    
    # Launch training processes
    try:
        mp.spawn(
            main,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        # Cleanup in case of error
        if dist.is_initialized():
            cleanup()
