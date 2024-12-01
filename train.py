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
from PIL import Image
import numpy as np
from models.attgan import Generator, Discriminator
from dataset import CustomDataset, get_transforms
from tqdm import tqdm
import pandas as pd
import time
from models.vgg_perceptual_loss import VGGPerceptualLoss
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from facenet_pytorch import InceptionResnetV1, MTCNN
import lpips
from torch.cuda.amp import GradScaler
import datetime
import json
from torch.optim.lr_scheduler import LambdaLR
import shutil
import traceback
import warnings
import torchvision

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Custom Loss Functions
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
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        # Apply Laplacian filter
        laplacian_kernel = self.laplacian_kernel.to(x.device)
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

# **Added Hinge Loss Class**
class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, logits, is_real):
        if is_real:
            # For real images, we want logits to be greater than or equal to 1
            loss = torch.mean(F.relu(1.0 - logits))
        else:
            # For fake images, we want logits to be less than or equal to -1
            loss = torch.mean(F.relu(1.0 + logits))
        return loss

def setup(rank, world_size):
    # Set environment variables for master address and port
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12356'

    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank
    )

    # Set device for this process
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def get_linear_warmup_scheduler(optimizer, warmup_epochs):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        return 1.0
    return LambdaLR(optimizer, lr_lambda)

class AttGANTrainer:
    def __init__(self, config, gpu_id):
        """Initialize trainer with proper use of all config parameters"""
        self.config = config
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(self.device)

        # Clear CUDA cache if configured
        if self.config['clear_cache_on_start']:
            torch.cuda.empty_cache()

        # Initialize models and move to device
        self.initialize_models()

        # Setup data loaders with proper normalization
        self.setup_dataloaders()

        # Initialize losses and metrics
        self.initialize_losses()

        # Initialize optimizers
        self.initialize_optimizers()

        # Initialize learning rate scheduler
        self.initialize_lr_scheduler()

        # Initialize mixed precision scaler
        self.scaler = GradScaler(enabled=self.config['use_amp'])

        # Initialize TensorBoard writer
        if self.gpu_id == 0:
            self.writer = SummaryWriter(self.config['log_dir'])

        # Load pretrained weights
        self.start_epoch = self.load_pretrained_weights()

        # Initialize best score for early stopping
        self.best_score = float('inf')
        self.patience_counter = 0

        # Initialize global step for logging
        self.global_step = 0

        # **Initialize Hinge Loss instead of BCE**
        self.adversarial_loss = HingeLoss()
        self.reconstruction_loss = nn.L1Loss()

    def initialize_models(self):
        """Initialize models with proper configuration"""
        # **Apply spectral normalization to the discriminator in the model definition**
        self.G = Generator().to(self.device)
        self.D = Discriminator().to(self.device)

        if self.config['world_size'] > 1:
            self.G = DDP(self.G, device_ids=[self.gpu_id], output_device=self.gpu_id)
            self.D = DDP(self.D, device_ids=[self.gpu_id], output_device=self.gpu_id)

    def setup_dataloaders(self):
        """Setup dataloaders with optimized loading parameters"""
        num_workers = self.config['num_workers'] if self.config['num_workers'] is not None else 0

        # **Added data augmentations in transforms**
        transform = get_transforms(
            image_size=self.config['image_size'],
            normalize=self.config['normalize_samples'],
            mean=self.config['normalization_mean'],
            std=self.config['normalization_std']
        )


        # In AttGANTrainer.setup_dataloaders()
        train_dataset = CustomDataset(
            self.config['dataset_path'],
            phase='train',
            transform=transform,
            augment=self.config.get('augment_synthetic_data', False)  # Pass augment flag here
        )

        val_dataset = CustomDataset(
            self.config['dataset_path'],
            phase='val',
            transform=transform,
            augment=False  # Do not augment validation data
        )



        # Set up samplers for distributed training
        train_sampler = DistributedSampler(train_dataset) if self.config['world_size'] > 1 else None
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if self.config['world_size'] > 1 else None

        # Configure DataLoader
        dataloader_kwargs = {
            'batch_size': self.config['batch_size'],
            'num_workers': num_workers,
            'pin_memory': self.config['dataloader']['pin_memory'],
            'drop_last': self.config['dataloader']['drop_last'],
            'persistent_workers': self.config['dataloader']['persistent_workers'],
            'prefetch_factor': self.config['prefetch_factor']
        }

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

    def initialize_losses(self):
        """Initialize losses with face recognition components"""
        # Face recognition model
        self.face_model = InceptionResnetV1(pretrained='vggface2', classify=False).eval().to(self.device)

        # Freeze face model parameters
        for param in self.face_model.parameters():
            param.requires_grad = False

        # MTCNN for face alignment (if needed)
        # self.mtcnn = MTCNN(device=self.device)

        # Perceptual loss
        self.perceptual_loss = VGGPerceptualLoss().to(self.device)

        # LPIPS loss
        self.lpips_loss = lpips.LPIPS(net='alex').to(self.device)

        # Sharpness and total variation loss
        self.sharpness_loss = SharpenessLoss().to(self.device)
        self.tv_loss = TotalVariationLoss().to(self.device)

        # FID metric
        self.fid = FrechetInceptionDistance(normalize=True).to(self.device)

    def initialize_optimizers(self):
        """Initialize optimizers"""
        self.g_optimizer = torch.optim.Adam(
            self.G.parameters(),
            lr=self.config['lr'],
            betas=(self.config['beta1'], self.config['beta2'])
        )
        self.d_optimizer = torch.optim.Adam(
            self.D.parameters(),
            lr=self.config['lr'],
            betas=(self.config['beta1'], self.config['beta2'])
        )

    # In initialize_lr_scheduler method
    def initialize_lr_scheduler(self):
        if self.config['use_lr_scheduler']:
            if self.config['lr_scheduler_type'] == 'StepLR':
                self.g_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.g_optimizer,
                    step_size=self.config['lr_step_size'],
                    gamma=self.config['lr_gamma']
                )
                self.d_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.d_optimizer,
                    step_size=self.config['lr_step_size'],
                    gamma=self.config['lr_gamma']
                )
        # Add other scheduler types if needed

    def load_pretrained_weights(self):
        """Load pretrained weights with comprehensive error handling"""
        if self.gpu_id == 0:
            print("\nLoading pretrained weights...")

        try:
            if self.config['resume_from_checkpoint']:
                checkpoint_path = self.config['resume_from_checkpoint']
                if os.path.exists(checkpoint_path):
                    return self.load_checkpoint(checkpoint_path)

            # No pretrained weights found
            if self.gpu_id == 0:
                print("\nNo pretrained weights found. Starting training from scratch...")
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

            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Load model states
            if isinstance(self.G, DDP):
                self.G.module.load_state_dict(checkpoint['G_state_dict'], strict=False)
                self.D.module.load_state_dict(checkpoint['D_state_dict'], strict=False)
            else:
                self.G.load_state_dict(checkpoint['G_state_dict'], strict=False)
                self.D.load_state_dict(checkpoint['D_state_dict'], strict=False)

            # Load optimizer states if available
            if 'g_optimizer' in checkpoint and 'd_optimizer' in checkpoint:
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
            for epoch in range(self.start_epoch, self.config['epochs']):
                if self.gpu_id == 0:
                    print(f"\nEpoch {epoch}/{self.config['epochs']}")
                self.train_loader.sampler.set_epoch(epoch) if self.config['world_size'] > 1 else None

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
        if self.gpu_id == 0 and self.config['use_tqdm']:
            pbar = tqdm(
                total=len(self.train_loader),
                desc=f'Epoch {epoch}/{self.config["epochs"]}',
                leave=True
            )

        # Training loop
        for batch_idx, (real_A, real_B) in enumerate(self.train_loader):
            try:
                # Training step
                g_loss, d_loss = self.training_step(real_A, real_B)

                # Save samples if needed
                if should_save_samples and batch_idx == 0:
                    self.save_samples(epoch, real_A, real_B)

                # Update progress
                if self.gpu_id == 0 and self.config['use_tqdm']:
                    if batch_idx % self.config['log_frequency'] == 0:
                        pbar.set_postfix({
                            'G_loss': f'{g_loss:.4f}',
                            'D_loss': f'{d_loss:.4f}'
                        })
                    pbar.update(1)

                self.global_step += 1

            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.handle_oom_error(batch_idx, e)
                    continue
                raise

        if self.gpu_id == 0 and self.config['use_tqdm']:
            pbar.close()

    def training_step(self, real_A, real_B):
        """Training step with gradient clipping"""
        try:
            # Ensure inputs are on correct device
            real_A = real_A.to(self.device, non_blocking=True)
            real_B = real_B.to(self.device, non_blocking=True)

            # Reset gradients
            self.g_optimizer.zero_grad(set_to_none=True)
            self.d_optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.config['use_amp']):
                # Generate fake images
                fake_B = self.process_large_image(real_A)

                # Train Discriminator
                d_loss = self.train_discriminator(real_B, fake_B.detach())

            # Discriminator backward pass
            self.scaler.scale(d_loss).backward()

            # Clip D gradients
            if self.config['gradient_clip_value'] > 0 and self.config['clip_discriminator']:
                self.scaler.unscale_(self.d_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.D.parameters(),
                    self.config['gradient_clip_value']
                )

            # Update discriminator weights
            self.scaler.step(self.d_optimizer)

            with torch.cuda.amp.autocast(enabled=self.config['use_amp']):
                # Train Generator
                fake_B = self.process_large_image(real_A)
                g_loss = self.train_generator(real_A, real_B, fake_B)

            # Generator backward pass
            self.scaler.scale(g_loss).backward()

            # Clip G gradients
            if self.config['gradient_clip_value'] > 0 and self.config['clip_generator']:
                self.scaler.unscale_(self.g_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.G.parameters(),
                    self.config['gradient_clip_value']
                )

            # Update generator weights
            self.scaler.step(self.g_optimizer)

            # Update scaler
            self.scaler.update()

            # Learning rate scheduler step
            if self.config['use_lr_scheduler']:
                self.g_scheduler.step()
                self.d_scheduler.step()

            # Log progress
            self.log_progress(g_loss.item(), d_loss.item())

            return g_loss.item(), d_loss.item()

        except Exception as e:
            self.handle_training_error(e)
            raise

    def process_large_image(self, image):
        """Process large images with consistent device handling"""
        B, C, H, W = image.shape
        chunk_size = self.config['process_chunk_size']

        # Ensure input is on correct device
        image = image.to(self.device, non_blocking=True)
        output = torch.zeros_like(image, device=self.device)

        if H <= chunk_size and W <= chunk_size:
            # Process entire image at once
            output = self.G(image)
        else:
            for h in range(0, H, chunk_size):
                for w in range(0, W, chunk_size):
                    h_end = min(h + chunk_size, H)
                    w_end = min(w + chunk_size, W)
                    chunk = image[:, :, h:h_end, w:w_end]

                    with torch.cuda.amp.autocast(enabled=self.config['use_amp']):
                        output[:, :, h:h_end, w:w_end] = self.G(chunk)

        return output

    def train_discriminator(self, real_B, fake_B):
        """Train discriminator with Hinge Loss"""
        # Real images
        real_validity = self.D(real_B)
        d_real_loss = self.adversarial_loss(real_validity, is_real=True)

        # Fake images
        fake_validity = self.D(fake_B)
        d_fake_loss = self.adversarial_loss(fake_validity, is_real=False)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        return d_loss

    def train_generator(self, real_A, real_B, fake_B):
        """Train generator with enhanced identity preservation"""
        # Reconstruction losses
        g_rec_loss = self.reconstruction_loss(fake_B, real_B)
        g_perceptual_loss = self.perceptual_loss(fake_B, real_B)

        # Adversarial loss
        fake_validity = self.D(fake_B)
        # **For generator, use adversarial loss as -mean(fake_validity)**
        g_adv_loss = -torch.mean(fake_validity)

        # Sharpness and TV loss
        g_sharpness_loss = self.sharpness_loss(fake_B)
        g_tv_loss = self.tv_loss(fake_B)

        # Face identity loss
        face_identity_loss = self.compute_face_identity_loss(real_B, fake_B)

        # Combined loss
        g_loss = (
            self.config['lambda_adv'] * g_adv_loss +
            self.config['lambda_rec'] * g_rec_loss +
            self.config['lambda_perceptual'] * g_perceptual_loss +
            self.config['lambda_identity'] * face_identity_loss +
            self.config['lambda_sharpness'] * g_sharpness_loss +
            self.config['lambda_tv'] * g_tv_loss
        )

        return g_loss

    def compute_face_identity_loss(self, real_images, fake_images):
        """Compute identity preservation loss using face recognition"""
        real_embeddings = self.face_model(real_images)
        fake_embeddings = self.face_model(fake_images)
        real_embeddings = F.normalize(real_embeddings, p=2, dim=1)
        fake_embeddings = F.normalize(fake_embeddings, p=2, dim=1)
        identity_loss = -F.cosine_similarity(real_embeddings, fake_embeddings).mean()
        return identity_loss

    def validate(self, epoch):
        """Validation with consistent device handling"""
        self.G.eval()
        metrics = {'val_g_loss': 0.0, 'val_fid': 0.0, 'val_lpips': 0.0, 'val_identity_sim': 0.0}
        num_batches = 0

        try:
            with torch.no_grad():
                for batch_idx, (real_A, real_B) in enumerate(self.val_loader):
                    real_A = real_A.to(self.device, non_blocking=True)
                    real_B = real_B.to(self.device, non_blocking=True)

                    fake_B = self.process_large_image(real_A)

                    # Compute validation losses
                    g_loss = self.reconstruction_loss(fake_B, real_B)
                    metrics['val_g_loss'] += g_loss.item()

                    # Compute FID
                    if batch_idx % self.config['validation']['fid_frequency'] == 0:
                        self.fid.update(real_B, real=True)
                        self.fid.update(fake_B, real=False)

                    # Compute LPIPS
                    lpips_distance = self.lpips_loss(real_B, fake_B).mean()
                    metrics['val_lpips'] += lpips_distance.item()

                    # Compute identity similarity
                    identity_similarity = self.compute_identity_similarity(real_B, fake_B)
                    metrics['val_identity_sim'] += identity_similarity

                    num_batches += 1

                    # Save validation images from main process
                    if batch_idx == 0 and self.gpu_id == 0:
                        self.save_validation_images(epoch, real_A, fake_B, real_B)

                # Finalize metrics
                metrics['val_g_loss'] /= num_batches
                metrics['val_lpips'] /= num_batches
                metrics['val_identity_sim'] /= num_batches

                # Compute FID
                metrics['val_fid'] = self.fid.compute().item()
                self.fid.reset()

                # Log validation metrics
                self.log_validation_metrics(epoch, metrics)

                # Early stopping check
                should_stop = self.early_stopping_check(epoch, metrics)
                return should_stop

        except Exception as e:
            self.handle_validation_error(e)
            return False

    def compute_identity_similarity(self, real_images, fake_images):
        """Compute cosine similarity between face embeddings"""
        real_embeddings = self.face_model(real_images)
        fake_embeddings = self.face_model(fake_images)
        real_embeddings = F.normalize(real_embeddings, p=2, dim=1)
        fake_embeddings = F.normalize(fake_embeddings, p=2, dim=1)
        similarity = F.cosine_similarity(real_embeddings, fake_embeddings).mean().item()
        return similarity

    def early_stopping_check(self, epoch, metrics):
        """Enhanced early stopping with multiple metrics"""
        metric_weights = self.config['validation']['metric_weights']
        current_score = (
            metric_weights['val_g_loss'] * metrics['val_g_loss'] +
            metric_weights['val_fid'] * metrics['val_fid'] +
            metric_weights['val_lpips'] * metrics['val_lpips'] -
            metric_weights['val_identity_sim'] * metrics['val_identity_sim']
        )

        if current_score < self.best_score:
            self.best_score = current_score
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

    def log_progress(self, g_loss, d_loss):
        """Enhanced progress logging"""
        if self.gpu_id == 0 and hasattr(self, 'writer'):
            self.writer.add_scalar('Train/G_loss', g_loss, self.global_step)
            self.writer.add_scalar('Train/D_loss', d_loss, self.global_step)

    def save_checkpoint(self, epoch, is_best=False, filename=None):
        """Save checkpoint with proper error handling and cleanup"""
        try:
            checkpoint_dir = self.config['checkpoint_dir']
            if filename is None:
                checkpoint_path = os.path.join(
                    checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'
                )
            else:
                checkpoint_path = os.path.join(checkpoint_dir, filename)

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

            # Save checkpoint
            torch.save(checkpoint, checkpoint_path)

            # Save best model if needed
            if is_best:
                best_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save(checkpoint, best_path)

            if self.gpu_id == 0:
                print(f"✓ Checkpoint saved: {checkpoint_path}")

        except Exception as e:
            if self.gpu_id == 0:
                print(f"❌ Error saving checkpoint: {str(e)}")

    def denormalize(self, tensor):
        """Denormalize tensor using mean and std from config"""
        mean = torch.tensor(self.config['normalization_mean']).view(-1, 1, 1).to(tensor.device)
        std = torch.tensor(self.config['normalization_std']).view(-1, 1, 1).to(tensor.device)
        return tensor * std + mean

    def save_samples(self, epoch, real_A, real_B):
        """Save sample images"""
        if self.gpu_id != 0:
            return

        try:
            sample_dir = self.config['sample_dir']
            os.makedirs(sample_dir, exist_ok=True)

            with torch.no_grad():
                fake_B = self.process_large_image(real_A)

            # Move tensors to CPU
            real_A_cpu = real_A.detach().cpu()
            fake_B_cpu = fake_B.detach().cpu()
            real_B_cpu = real_B.detach().cpu()

            # Denormalize images
            real_A_cpu = self.denormalize(real_A_cpu)
            fake_B_cpu = self.denormalize(fake_B_cpu)
            real_B_cpu = self.denormalize(real_B_cpu)

            # Concatenate images
            img_sample = torch.cat((real_A_cpu, fake_B_cpu, real_B_cpu), dim=0)

            save_path = os.path.join(sample_dir, f'epoch_{epoch}.png')
            save_image(img_sample, save_path, nrow=real_A_cpu.size(0), normalize=False)

            # Log images to TensorBoard
            if hasattr(self, 'writer'):
                grid = torchvision.utils.make_grid(img_sample, nrow=real_A_cpu.size(0))
                self.writer.add_image('Samples', grid, epoch)

        except Exception as e:
            print(f"Error saving samples: {str(e)}")

    def save_validation_images(self, epoch, real_A, fake_B, real_B):
        """Save sample validation images"""
        if self.gpu_id != 0:
            return

        try:
            # Move tensors to CPU
            real_A_cpu = real_A.detach().cpu()
            fake_B_cpu = fake_B.detach().cpu()
            real_B_cpu = real_B.detach().cpu()

            # Denormalize images
            real_A_cpu = self.denormalize(real_A_cpu)
            fake_B_cpu = self.denormalize(fake_B_cpu)
            real_B_cpu = self.denormalize(real_B_cpu)

            # Concatenate images
            img_sample = torch.cat((real_A_cpu, fake_B_cpu, real_B_cpu), dim=0)

            save_path = os.path.join(self.config['sample_dir'], f'validation_epoch_{epoch}.png')
            save_image(img_sample, save_path, nrow=real_A_cpu.size(0), normalize=False)

            # Log images to TensorBoard
            if hasattr(self, 'writer'):
                grid = torchvision.utils.make_grid(img_sample, nrow=real_A_cpu.size(0))
                self.writer.add_image('Validation/Samples', grid, epoch)

        except Exception as e:
            print(f"Error saving validation images: {str(e)}")

    def log_validation_metrics(self, epoch, metrics):
        """Enhanced validation metrics logging"""
        stats = {
            'epoch': epoch,
            'timestamp': datetime.datetime.now().isoformat(),
            **metrics,
            'best_score': self.best_score,
            'patience_counter': self.patience_counter
        }

        # Save to CSV
        df = pd.DataFrame([stats])
        log_file = os.path.join(self.config['log_dir'], 'validation_metrics.csv')
        df.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)

        # Log to TensorBoard
        if hasattr(self, 'writer'):
            for key, value in metrics.items():
                self.writer.add_scalar(f'Validation/{key}', value, epoch)

    def handle_oom_error(self, identifier, error_msg):
        """Handle out of memory errors with recovery attempts"""
        if self.gpu_id == 0:
            print(f"\nOOM error at {identifier}: {error_msg}")
            print("Attempting recovery...")

        torch.cuda.empty_cache()

    def handle_training_error(self, error):
        """Handle training errors"""
        if self.gpu_id == 0:
            print(f"\nTraining error: {str(error)}")

    def handle_validation_error(self, error):
        """Handle validation errors"""
        if self.gpu_id == 0:
            print(f"\nValidation error: {str(error)}")

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

            error_log_path = os.path.join(self.config['log_dir'], 'critical_errors.json')
            with open(error_log_path, 'a') as f:
                json.dump(error_info, f)
                f.write('\n')

            if hasattr(self, 'writer'):
                self.writer.add_text('critical_error', str(error_info))

def main(rank, world_size, config):
    try:
        # Update config with process-specific info
        config['rank'] = rank
        config['world_size'] = world_size

        setup(rank, world_size)

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
        'batch_size': 2,          # Increased batch size
        'num_workers': 4,
        'lr': 1e-5,  # Reduce from 1e-4 to 1e-5
        'beta1': 0.5,
        'beta2': 0.999,
        'epochs': 200,

        # Model parameters
        'image_size': 512,        # Reduced image size
        'lambda_adv': 2.0,          # Increase from 1.0 to 2.0
        'lambda_rec': 50.0,         # Keep the same or adjust slightly
        'lambda_perceptual': 10.0,  # Increase from 5.0 to 10.0
        'lambda_identity': 5.0,     # Increase from 2.0 to 5.0
        'lambda_sharpness': 2.0,    # Increase from 1.0 to 2.0
        'lambda_tv': 0.0,           # Keep as is unless you want to penalize total variation

        # Memory optimization
        'use_amp': True,
        'gradient_accumulation_steps': 1,  # No accumulation needed with increased batch size
        'process_chunk_size': 512,         # Can process the whole image at once

        # Training schedule
        'val_frequency': 1,
        'save_frequency': 5,

        # Distributed training
        'world_size': torch.cuda.device_count(),

        # Learning rate scheduling
        'use_lr_scheduler': True,
        'lr_scheduler_type': 'StepLR',
        'lr_step_size': 30,   # Decrease LR every 30 epochs
        'lr_gamma': 0.5,      # Reduce LR by half
        'warmup_epochs': 5,

        # Checkpoint handling
        'resume_from_checkpoint': None,
        'save_on_error': True,
        'clear_cache_on_start': True,

        # Image normalization
        'normalize_samples': True,
        # Updated normalization parameters
        'normalization_mean': [0.5, 0.5, 0.5],
        'normalization_std': [0.5, 0.5, 0.5],

        # Early stopping
        'early_stopping_patience': 30,

        # DataLoader settings
        'prefetch_factor': 2,
        'dataloader': {
            'pin_memory': True,
            'drop_last': True,
            'persistent_workers': True
        },

        # Sample saving
        'sample_frequency': 2,
        'initial_sample_epochs': 5,
        'final_sample_epochs': 5,
        'keep_last_samples': 5,

        # Validation metrics
        'validation': {
            'fid_frequency': 5,
            'metric_weights': {
                'val_g_loss': 0.2,
                'val_fid': 0.4,
                'val_lpips': 0.2,
                'val_identity_sim': 0.2
            }
        },

        # Logging
        'use_tqdm': True,
        'log_frequency': 50,
        'dataset_path': 'datasets/dataset',

        # Gradient clipping
        'gradient_clip_value': 1.0,
        'gradient_clip_algorithm': 'norm',
        'clip_discriminator': True,
        'clip_generator': True,
        'augment_synthetic_data': True,

        # Additional configurations
        'enable_gradient_checkpointing': False,  # Not needed with lower resolution
    }

    # Create necessary directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['sample_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)

    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    world_size = config['world_size']

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
        if dist.is_initialized():
            cleanup()
