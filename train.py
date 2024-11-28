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

    def setup_dataloaders(self):
        transform = transforms.Compose([
            transforms.Resize((self.config['image_size'], self.config['image_size']), 
                            interpolation=transforms.InterpolationMode.BILINEAR),
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
        for epoch in range(self.config['epochs']):
            self.train_loader.sampler.set_epoch(epoch)
            self.train_epoch(epoch)
            
            if epoch % self.config['val_frequency'] == 0:
                self.validate(epoch)
            
            # Save checkpoints (only from main process)
            if self.gpu_id == 0 and epoch % self.config['save_frequency'] == 0:
                self.save_checkpoint(epoch)

    def process_large_image(self, image):
        """Process large images in chunks"""
        B, C, H, W = image.shape
        chunk_size = self.config['batch_chunk_size']
        
        if H <= chunk_size and W <= chunk_size:
            return self.G(image)
            
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
        
        for batch_idx, (real_A, real_B) in enumerate(tqdm(self.train_loader)):
            real_A = real_A.to(self.device)
            real_B = real_B.to(self.device)
            
            # Train Discriminator
            self.d_optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.config['use_amp']):
                fake_B = self.G(real_A)
                d_loss = self.train_discriminator(real_B, fake_B)
                
            self.scaler.scale(d_loss).backward()
            self.scaler.step(self.d_optimizer)
            
            # Train Generator
            self.g_optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.config['use_amp']):
                g_loss = self.train_generator(real_A, real_B, fake_B)
                
            self.scaler.scale(g_loss).backward()
            self.scaler.step(self.g_optimizer)
            self.scaler.update()
            
            if batch_idx % self.config['log_frequency'] == 0:
                self.log_progress(epoch, batch_idx, g_loss, d_loss)

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

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'G_state_dict': self.G.module.state_dict(),
            'D_state_dict': self.D.module.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'best_fid': self.best_fid,
        }
        
        # Save regular checkpoint
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
        """Train discriminator"""
        batch_size = real_B.size(0)
        
        # Real images
        real_validity = self.D(real_B)
        real_labels = torch.ones_like(real_validity).to(self.device)
        d_real_loss = self.adversarial_loss(real_validity, real_labels)
        
        # Fake images
        fake_validity = self.D(fake_B.detach())
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
        
        # Reconstruction loss
        g_rec_loss = self.reconstruction_loss(fake_B, real_B)
        
        # Perceptual loss
        g_perceptual_loss = self.perceptual_loss(fake_B, real_B)
        
        # Identity loss
        g_identity_loss = self.identity_loss(fake_B, real_B)
        
        # Total generator loss
        g_loss = (
            self.config['lambda_adv'] * g_adv_loss +
            self.config['lambda_rec'] * g_rec_loss +
            self.config['lambda_perceptual'] * g_perceptual_loss +
            self.config['lambda_identity'] * g_identity_loss
        )
        
        return g_loss

def setup(rank, world_size, config):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group with timeout
    dist.init_process_group(
        backend='nccl',  # Use NCCL backend directly
        init_method='tcp://127.0.0.1:12355',  # Use explicit IP instead of localhost
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(minutes=30)  # Add timeout
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
        
        # Training parameters
        'batch_size': 8,          # Increased from 1, but still reasonable for 512x512
        'num_workers': 4,         # 2 * num_gpus is a good rule of thumb
        'lr': 0.00002 * (8/1),   # Scale learning rate with batch size
        'beta1': 0.5,
        'beta2': 0.999,
        'epochs': 200,           # Can reduce epochs with larger batch size
        
        # Model parameters - Tuned for augmented pairs
        'image_size': 512,       # Keep 512 for detail preservation
        'lambda_adv': 0.03,      # Reduced further due to augmented variations
        'lambda_rec': 45.0,      # Increased to maintain consistency across augmentations
        'lambda_perceptual': 25.0,  # Increased to maintain perceptual consistency
        'lambda_identity': 20.0,    # Increased to ensure identity preservation across augmentations
        
        # Memory optimization
        'use_amp': True,         # Use automatic mixed precision
        'gradient_accumulation_steps': 1,  # Reduced since we increased batch size
        'batch_chunk_size': 64,   # Reduced to handle augmented data better
        
        # Training schedule
        'val_frequency': 1,      # Keep frequent validation
        'save_frequency': 3,     # More frequent saving due to augmented data
        'log_frequency': 5,      # More frequent logging to track augmented variations
        
        # Distributed training
        'dist_url': 'tcp://127.0.0.1:12355',
        'dist_backend': 'nccl',
        'multiprocessing_distributed': True,
        'pin_memory': True,       # Enable pin memory for faster data transfer
        'prefetch_factor': 2,     # Reduced prefetch factor
        'use_lr_scheduler': True,
        'lr_scheduler_patience': 10,
        'lr_scheduler_factor': 0.5,
        
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
