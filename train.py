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

class AttGANTrainer:
    def __init__(self, config, gpu_id):
        self.gpu_id = gpu_id
        self.config = config
        
        # Setup device
        torch.cuda.set_device(gpu_id)
        self.device = torch.device(f'cuda:{gpu_id}')
        
        # Create models
        self.G = Generator().to(self.device)
        self.D = Discriminator().to(self.device)
        
        # Wrap models with DDP
        self.G = DDP(self.G, device_ids=[gpu_id])
        self.D = DDP(self.D, device_ids=[gpu_id])
        
        # Setup optimizers
        self.g_optimizer = torch.optim.Adam(
            self.G.parameters(), 
            lr=config['lr'], 
            betas=(config['beta1'], config['beta2'])
        )
        self.d_optimizer = torch.optim.Adam(
            self.D.parameters(), 
            lr=config['lr'], 
            betas=(config['beta1'], config['beta2'])
        )
        
        # Setup learning rate schedulers if enabled
        if config.get('use_lr_scheduler', False):  # Using .get() with default
            self.g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.g_optimizer, 
                mode='min', 
                patience=config['lr_scheduler_patience'],
                factor=config['lr_scheduler_factor']
            )
            self.d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.d_optimizer,
                mode='min',
                patience=config['lr_scheduler_patience'],
                factor=config['lr_scheduler_factor']
            )
        
        # Setup automatic mixed precision
        self.scaler = GradScaler(enabled=config['use_amp'])
        
        # Setup losses
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.reconstruction_loss = nn.L1Loss()
        
        # Setup tensorboard (only for main process)
        if self.gpu_id == 0:
            self.writer = SummaryWriter(config['log_dir'])
        
        # Setup data loaders
        self.setup_dataloaders()
        
        # Add perceptual loss
        self.perceptual_loss = VGGPerceptualLoss().to(self.device)
        
        # Add identity loss
        self.identity_loss = nn.L1Loss()
        
        # Initialize FID metric only if torch-fidelity is available
        try:
            self.fid = FrechetInceptionDistance(normalize=True).to(self.device)
            self.use_fid = True
        except ModuleNotFoundError:
            print("Warning: torch-fidelity not installed. FID calculation will be disabled.")
            self.use_fid = False
        
        # Initialize LPIPS for perceptual similarity
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        
        # Initialize face recognition model
        self.face_model = InceptionResnetV1(pretrained='vggface2').to(self.device)
        self.face_model.eval()
        
        # Initialize face detection for alignment
        self.mtcnn = MTCNN(device=self.device)
        
        # Add validation metrics to track
        self.best_fid = float('inf')
        self.best_identity_score = 0.0
        
        # Add sharpness loss
        self.sharpness_loss = SharpenessLoss().to(self.device)
        
        # Add total variation loss to reduce artifacts
        self.tv_loss = TotalVariationLoss()
        
        # Add gradient clipping
        self.gradient_clip_value = config['gradient_clip_value']
        
        self.best_loss = float('inf')
        self.patience_counter = 0

    def setup_dataloaders(self):
        transform = transforms.Compose([
            transforms.Resize((self.config['image_size'], self.config['image_size']), 
                            interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Add dataset validation
        if self.gpu_id == 0:  # Only print on main process
            print(f"\nValidating dataset at {self.config['dataset_path']}")
            for split in ['train', 'val']:
                for domain in ['A', 'B']:
                    path = os.path.join(self.config['dataset_path'], f'{split}_{domain}')
                    if not os.path.exists(path):
                        raise RuntimeError(f"Dataset directory not found: {path}")
                    files = os.listdir(path)
                    print(f"Found {len(files)} images in {split}_{domain}")
                    if len(files) == 0:
                        raise RuntimeError(f"No images found in {path}")
        
        # Create datasets
        train_dataset = CustomDataset(
            root_dir=self.config['dataset_path'],
            phase='train',
            transform=transform
        )
        
        val_dataset = CustomDataset(
            root_dir=self.config['dataset_path'],
            phase='val',
            transform=transform
        )
        
        # Print dataset sizes
        if self.gpu_id == 0:
            print(f"\nDataset sizes:")
            print(f"Training: {len(train_dataset)} pairs")
            print(f"Validation: {len(val_dataset)} pairs")
        
        # Create samplers for distributed training
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=self.gpu_id
        )
        
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=dist.get_world_size(),
            rank=self.gpu_id
        )
        
        # Create data loaders with error handling
        try:
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['batch_size'],
                sampler=train_sampler,
                num_workers=self.config['num_workers'],
                pin_memory=self.config['pin_memory'],
                prefetch_factor=self.config['prefetch_factor'],
                drop_last=True
            )
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                sampler=val_sampler,
                num_workers=self.config['num_workers'],
                pin_memory=self.config['pin_memory'],
                prefetch_factor=self.config['prefetch_factor']
            )
            
            # Verify we can load at least one batch
            if self.gpu_id == 0:
                print("\nVerifying data loading...")
                train_iter = iter(self.train_loader)
                first_batch = next(train_iter)
                print(f"Successfully loaded first batch of size: {first_batch[0].shape}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to create data loaders on GPU {self.gpu_id}: {str(e)}")

    def train(self):
        """Main training loop with proper epoch counting and resume capability"""
        try:
            # Try to load progress from JSON first
            progress_file = os.path.join(self.config['checkpoint_dir'], 'training_progress.json')
            if os.path.exists(progress_file):
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    start_epoch = progress.get('current_epoch', 0)
                    self.best_fid = progress.get('best_fid', float('inf'))
                    print(f"Resuming from epoch {start_epoch}")
            else:
                # If no progress file, try to load from latest checkpoint
                start_epoch = self.load_checkpoint(os.path.join(
                    self.config['checkpoint_dir'], 
                    'latest.pth'
                ))

            # Main training loop
            for epoch in range(start_epoch, self.config['epochs']):
                try:
                    print(f"\nStarting epoch {epoch}")
                    self.train_epoch(epoch)
                    
                    # Validation step
                    if epoch % self.config['val_frequency'] == 0:
                        self.validate(epoch)
                    
                    # Save checkpoint with correct epoch number
                    if epoch % self.config['save_frequency'] == 0:
                        self.save_checkpoint(epoch)
                    
                    # Always save latest checkpoint and progress
                    self.save_checkpoint(epoch, filename='latest.pth')
                    self.save_training_progress(epoch)
                    
                except Exception as e:
                    print(f"Error during epoch {epoch}: {str(e)}")
                    # Save checkpoint on error
                    self.save_checkpoint(epoch, filename=f'error_epoch_{epoch}.pth')
                    raise e

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            # Save checkpoint on interrupt
            self.save_checkpoint(epoch, filename='interrupt.pth')
            
        except Exception as e:
            print(f"\nTraining failed with error: {str(e)}")
            # Save checkpoint on error
            if 'epoch' in locals():
                self.save_checkpoint(epoch, filename='error.pth')
            raise e

    def process_large_image(self, image):
        """Process large images in smaller chunks"""
        B, C, H, W = image.shape
        chunk_size = self.config['process_chunk_size']
        
        if H <= chunk_size and W <= chunk_size:
            return image
            
        # Process image in chunks
        output = torch.zeros_like(image)
        for h in range(0, H, chunk_size):
            for w in range(0, W, chunk_size):
                h_end = min(h + chunk_size, H)
                w_end = min(w + chunk_size, W)
                chunk = image[:, :, h:h_end, w:w_end]
                
                with torch.cuda.amp.autocast(enabled=self.config['use_amp']):
                    output[:, :, h:h_end, w:w_end] = self.G(chunk)
                    
        return output

    def train_epoch(self, epoch):
        self.G.train()
        self.D.train()
        
        # Clear cache at start of epoch
        torch.cuda.empty_cache()
        
        for batch_idx, (real_A, real_B) in enumerate(tqdm(self.train_loader)):
            try:
                # Process images in chunks if needed
                real_A = self.process_large_image(real_A.to(self.device))
                real_B = self.process_large_image(real_B.to(self.device))
                
                # Zero gradients
                self.d_optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                self.g_optimizer.zero_grad(set_to_none=True)
                
                # Accumulate gradients
                d_loss_acc = 0
                g_loss_acc = 0
                
                for acc_step in range(self.config['gradient_accumulation_steps']):
                    with torch.cuda.amp.autocast(enabled=self.config['use_amp']):
                        # Generate fake images in chunks if needed
                        fake_B = self.process_large_image(self.G(real_A))
                        
                        # Train Discriminator
                        d_loss = self.train_discriminator(real_B, fake_B.detach())
                        d_loss = d_loss / self.config['gradient_accumulation_steps']
                        d_loss_acc += d_loss.item()
                    
                    # Backward pass for discriminator
                    self.scaler.scale(d_loss).backward()
                    
                    with torch.cuda.amp.autocast(enabled=self.config['use_amp']):
                        # Train Generator
                        g_loss = self.train_generator(real_A, real_B, fake_B)
                        g_loss = g_loss / self.config['gradient_accumulation_steps']
                        g_loss_acc += g_loss.item()
                    
                    # Backward pass for generator
                    self.scaler.scale(g_loss).backward()
                    
                    # Clear some memory
                    del fake_B
                    torch.cuda.empty_cache()
                
                # Update weights
                self.scaler.step(self.d_optimizer)
                self.scaler.step(self.g_optimizer)
                self.scaler.update()
                
                # Clear memory every batch
                if batch_idx % self.config['empty_cache_frequency'] == 0:
                    torch.cuda.empty_cache()
                
                # Log with accumulated loss
                if batch_idx % self.config['log_frequency'] == 0:
                    print(f'[GPU {self.gpu_id}] '
                          f'[Epoch {epoch}/{self.config["epochs"]}] '
                          f'[Batch {batch_idx}/{len(self.train_loader)}] '
                          f'[D loss: {d_loss_acc:.4f}] '
                          f'[G loss: {g_loss_acc:.4f}]')
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"WARNING: out of memory on batch {batch_idx}. Skipping batch")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e

    def log_progress(self, epoch, batch_idx, g_loss, d_loss):
        """Log training progress"""
        # Only log to tensorboard from the main process
        if self.gpu_id == 0 and hasattr(self, 'writer'):
            # Log to tensorboard
            step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/G_loss', g_loss.item(), step)
            self.writer.add_scalar('Train/D_loss', d_loss.item(), step)
            
            # Log learning rates
            self.writer.add_scalar('LR/Generator', 
                                self.g_optimizer.param_groups[0]['lr'], step)
            self.writer.add_scalar('LR/Discriminator', 
                                self.d_optimizer.param_groups[0]['lr'], step)
        
        # Print progress (from all processes)
        print(f"[GPU {self.gpu_id}] [Epoch {epoch}/{self.config['epochs']}] "
              f"[Batch {batch_idx}/{len(self.train_loader)}] "
              f"[D loss: {d_loss.item():.4f}] "
              f"[G loss: {g_loss.item():.4f}]")

    def validate(self, epoch):
        self.G.eval()
        with torch.no_grad():
            for batch_idx, (real_A, real_B) in enumerate(self.val_loader):
                real_A = real_A.to(self.device)
                real_B = real_B.to(self.device)
                fake_B = self.G(real_A)
                
                # Save sample images
                if batch_idx == 0 and self.gpu_id == 0:
                    self.save_validation_images(epoch, real_A, fake_B, real_B)

    def log_validation_metrics(self, epoch, metrics):
        """Log validation metrics to CSV"""
        stats = {
            'epoch': epoch,
            'timestamp': time.time(),
            **metrics
        }
        
        df = pd.DataFrame([stats])
        df.to_csv('validation_metrics.csv', 
                 mode='a', 
                 header=not os.path.exists('validation_metrics.csv'))

    def save_checkpoint(self, epoch, is_best=False, filename=None):
        """Save checkpoint with the actual epoch number and optional custom filename"""
        checkpoint = {
            'epoch': epoch,
            'G_state_dict': self.G.module.state_dict() if isinstance(self.G, DDP) else self.G.state_dict(),
            'D_state_dict': self.D.module.state_dict() if isinstance(self.D, DDP) else self.D.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
        }
        
        if filename:
            # Use custom filename if provided
            checkpoint_path = os.path.join(self.config['checkpoint_dir'], filename)
        else:
            # Use default epoch-based filename
            checkpoint_path = os.path.join(
                self.config['checkpoint_dir'], 
                f'checkpoint_epoch_{epoch}.pth'
            )
        
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if needed
        if is_best:
            best_path = os.path.join(
                self.config['checkpoint_dir'], 
                'best_model.pth'
            )
            torch.save(checkpoint, best_path)
        
        # Remove old checkpoints to save space (optional)
        if not filename:  # Only cleanup numbered checkpoints
            self.cleanup_old_checkpoints(keep_last=5)

    def cleanup_old_checkpoints(self, keep_last=5):
        """Keep only the last N checkpoints"""
        checkpoint_dir = self.config['checkpoint_dir']
        
        # Ensure directory exists
        if not os.path.exists(checkpoint_dir):
            return
        
        try:
            # Get list of checkpoint files
            checkpoints = sorted([
                f for f in os.listdir(checkpoint_dir) 
                if f.startswith('checkpoint_epoch_') and f.endswith('.pth')
            ])
            
            # If we have more checkpoints than we want to keep
            if len(checkpoints) > keep_last:
                # Remove old checkpoints, keeping the last N
                for checkpoint in checkpoints[:-keep_last]:
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
                    if os.path.exists(checkpoint_path):  # Double check file exists
                        try:
                            os.remove(checkpoint_path)
                        except OSError as e:
                            print(f"Warning: Could not remove checkpoint {checkpoint}: {e}")
                            
        except Exception as e:
            print(f"Warning: Error during checkpoint cleanup: {e}")
            # Continue training even if cleanup fails
            pass

    def save_validation_images(self, epoch, real_A, fake_B, real_B):
        """Save sample validation images"""
        # Denormalize images from [-1, 1] to [0, 1]
        real_A = self.denormalize(real_A)
        fake_B = self.denormalize(fake_B)
        real_B = self.denormalize(real_B)
        
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
            normalize=True  # Only normalize, no range
        )
        
        # Log images to tensorboard if enabled
        if hasattr(self, 'writer'):
            self.writer.add_image(
                'Validation/real_A_fake_B_real_B', 
                img_sample[0],  # Take first image from batch
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
            distance = self.lpips_fn(real_images, fake_images)
            return distance.mean().item()

    def clear_gpu_memory(self):
        """Clear GPU memory cache"""
        torch.cuda.empty_cache()
    
    def get_gpu_memory_info(self):
        """Get GPU memory usage"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2, torch.cuda.memory_reserved() / 1024**2

    def train_discriminator(self, real_B, fake_B):
        """Train discriminator with detached fake images"""
        # Real images
        real_validity = self.D(real_B)
        real_labels = torch.ones_like(real_validity).to(self.device)
        d_real_loss = self.adversarial_loss(real_validity, real_labels)
        
        # Fake images
        fake_validity = self.D(fake_B.detach())  # Important: detach fake_B
        fake_labels = torch.zeros_like(fake_validity).to(self.device)
        d_fake_loss = self.adversarial_loss(fake_validity, fake_labels)
        
        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        return d_loss

    def train_generator(self, real_A, real_B, fake_B):
        """Train generator"""
        # Adversarial loss
        fake_validity = self.D(fake_B)
        real_labels = torch.ones_like(fake_validity).to(self.device)
        g_adv_loss = self.adversarial_loss(fake_validity, real_labels)
        
        # Other losses
        g_rec_loss = self.reconstruction_loss(fake_B, real_B)
        g_perceptual_loss = self.perceptual_loss(fake_B, real_B)
        g_identity_loss = self.identity_loss(fake_B, real_B)
        g_sharpness_loss = self.sharpness_loss(fake_B)
        g_tv_loss = self.tv_loss(fake_B)
        
        # Combine losses
        g_loss = (
            self.config['lambda_adv'] * g_adv_loss +
            self.config['lambda_rec'] * g_rec_loss +
            self.config['lambda_perceptual'] * g_perceptual_loss +
            self.config['lambda_identity'] * g_identity_loss +
            self.config['lambda_sharpness'] * g_sharpness_loss +
            self.config['lambda_tv'] * g_tv_loss
        )
        
        return g_loss

    def load_checkpoint(self, checkpoint_path):
        """Enhanced checkpoint loading with better error handling"""
        try:
            if os.path.exists(checkpoint_path):
                print(f"Loading checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # Load model states
                self.G.module.load_state_dict(checkpoint['G_state_dict']) if isinstance(self.G, DDP) \
                    else self.G.load_state_dict(checkpoint['G_state_dict'])
                self.D.module.load_state_dict(checkpoint['D_state_dict']) if isinstance(self.D, DDP) \
                    else self.D.load_state_dict(checkpoint['D_state_dict'])
                
                # Load optimizer states
                self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
                self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
                
                # Get epoch number
                start_epoch = checkpoint.get('epoch', 0) + 1  # Start from next epoch
                
                print(f"Successfully loaded checkpoint from epoch {start_epoch-1}")
                return start_epoch
                
            else:
                print("No checkpoint found, starting from scratch")
                return 0
                
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from scratch")
            return 0

    def log_training_progress(self, epoch, g_loss, d_loss):
        """Log training progress to CSV with correct epoch numbers"""
        stats = {
            'epoch': epoch,
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'timestamp': time.time(),
            'learning_rate': self.g_optimizer.param_groups[0]['lr']
        }
        
        # Save to CSV
        df = pd.DataFrame([stats])
        log_file = 'training_log.csv'
        df.to_csv(log_file, 
                  mode='a', 
                  header=not os.path.exists(log_file),
                  index=False)
        
        # Also save current epoch to a JSON file for easy recovery
        progress = {
            'current_epoch': epoch,
            'best_fid': self.best_fid,
            'last_update': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        with open('training_progress.json', 'w') as f:
            json.dump(progress, f, indent=4)

    def save_training_progress(self, epoch):
        """Save training progress to JSON"""
        progress = {
            'current_epoch': epoch + 1,  # Save next epoch to resume from
            'best_fid': self.best_fid,
            'last_update': time.strftime('%Y-%m-%d %H:%M:%S'),
            'training_params': {
                'learning_rate': self.g_optimizer.param_groups[0]['lr'],
                'batch_size': self.config['batch_size'],
                'image_size': self.config['image_size']
            }
        }
        
        progress_file = os.path.join(self.config['checkpoint_dir'], 'training_progress.json')
        try:
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not save progress file: {e}")

def setup(rank, world_size, config):
    os.environ['MASTER_PORT'] = '12356'
    # or
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://localhost:12356',
        world_size=world_size,
        rank=rank
    )
    
    # Set device for this process
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, config):
    try:
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
        'batch_size': 1,                    # Reduced from 2
        'num_workers': 1,                   # Reduce worker threads
        'lr': 0.00001,            # Keep low learning rate for stability
        'beta1': 0.5,
        'beta2': 0.999,
        'epochs': 200,
        
        # Model parameters - Tuned for high resolution
        'image_size': 1024,        # Increased resolution
        'lambda_adv': 0.01,        # Reduced for better stability at high resolution
        'lambda_rec': 45.0,        # Increased reconstruction weight
        'lambda_perceptual': 10.0, # Reduced to prevent over-emphasis on perceptual features
        'lambda_identity': 25.0,   # Increased for better identity preservation
        'lambda_sharpness': 0.8,   # Reduced to prevent over-sharpening at high res
        'lambda_tv': 0.05,         # Reduced TV loss for high res
        
        # Memory optimization - Critical for 1024x1024
        'use_amp': True,           # Keep AMP enabled
        'gradient_accumulation_steps': 8,   # Increased from 4
        'batch_chunk_size': 256,            # Increased for better memory efficiency
        
        # Training schedule
        'val_frequency': 1,
        'save_frequency': 2,       # More frequent saving
        'log_frequency': 10,       # Reduced logging frequency
        
        # Distributed training
        'dist_url': 'tcp://127.0.0.1:12356',
        'dist_backend': 'nccl',
        'multiprocessing_distributed': True,
        'pin_memory': True,
        'prefetch_factor': 1,              # Minimum prefetch
        
        # Learning rate scheduling
        'use_lr_scheduler': True,
        'lr_scheduler_patience': 8,
        'lr_scheduler_factor': 0.5,
        
        # Checkpoint handling
        'resume_from_checkpoint': None,
        'keep_last_checkpoints': 3,  # Reduced due to larger file sizes
        'save_best_model': True,
        
        # Training stability
        'gradient_clip_value': 0.5,  # Reduced for stability
        
        # Early stopping
        'early_stopping_patience': 15,
        'resume_training': True,
        
        # Memory cleanup
        'empty_cache_frequency': 1,         # Clear cache more frequently
        'process_chunk_size': 512,         # New parameter for image processing
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
