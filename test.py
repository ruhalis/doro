import torch
from torchvision import transforms
from PIL import Image
import os
from models.attgan import Generator
import argparse
from tqdm import tqdm
import gc  # for garbage collection
import numpy as np
import torch.nn.functional as F
import torchvision.utils as vutils

class Tester:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        # Check CUDA availability
        if 'cuda' in str(self.device):
            if torch.cuda.is_available():
                print(f"GPU: {torch.cuda.get_device_name(self.device)}")
                print(f"Memory Allocated: {torch.cuda.memory_allocated(self.device)/1024**2:.2f}MB")
            else:
                print("CUDA is not available. Falling back to CPU")
                self.device = torch.device('cpu')
        
        # Initialize generator
        self.G = Generator().to(self.device)
        self.G.eval()
        
        # Load checkpoint with error handling
        self.load_checkpoint(checkpoint_path)
        
        # Setup transforms
        self.image_size = 512  # Adjust based on your model's expected input size

        # Updated normalization to handle RGB images
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        
        # Updated inverse transform to correctly denormalize RGB images
        self.inverse_transform = transforms.Compose([
            transforms.Normalize(
                mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5], 
                std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
            ),
            transforms.Lambda(lambda x: x.clamp(0, 1)),
            transforms.ToPILImage()
        ])

    @torch.no_grad()
    def process_image(self, image_path, output_path=None):
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Apply transform
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Generate result
        with torch.cuda.amp.autocast(enabled=True):
            fake_img = self.G(img_tensor)
        
        # Move tensor to CPU and apply inverse transform
        fake_img_cpu = fake_img.detach().cpu().squeeze(0)
        output_img = self.inverse_transform(fake_img_cpu)
        
        # Save or return the output image
        if output_path:
            output_img.save(output_path)
            return True
        else:
            return output_img

    def clear_gpu_memory(self):
        """Clear GPU memory cache"""
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
            gc.collect()

    def process_batch(self, image_paths, output_dir):
        # Process multiple images at once
        for image_path in tqdm(image_paths, desc="Processing images"):
            output_path = os.path.join(output_dir, 'result_' + os.path.basename(image_path))
            success = self.process_image(image_path, output_path)
            if not success:
                print(f"Failed to process {image_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint with proper error handling"""
        try:
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state dict
            if 'G_state_dict' in checkpoint:
                self.G.load_state_dict(checkpoint['G_state_dict'])
            else:
                self.G.load_state_dict(checkpoint)
            
            print("Model loaded successfully")
            self.G.eval()  # Set to evaluation mode
            
        except Exception as e:
            print(f"Failed to load checkpoint: {str(e)}")
            print("\nTroubleshooting steps:")
            print("1. Verify the checkpoint file exists and is not corrupted")
            print("2. Check if the checkpoint was saved properly during training")
            print("3. Try using an earlier checkpoint")
            print("4. Verify the model architecture matches the checkpoint")
            raise RuntimeError(f"Failed to load checkpoint: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Test AttGAN on images')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output', type=str, required=True, help='Path to output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize tester
    tester = Tester(args.checkpoint, args.device)
    
    # Process images
    if os.path.isfile(args.input):
        # Single image
        output_path = os.path.join(args.output, 'result_' + os.path.basename(args.input))
        success = tester.process_image(args.input, output_path)
        if success:
            print(f"Processed image saved to {output_path}")
        else:
            print(f"Failed to process {args.input}")
    else:
        # Directory of images
        image_files = [os.path.join(args.input, f) for f in os.listdir(args.input) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Found {len(image_files)} images to process")
        
        tester.process_batch(image_files, args.output)

if __name__ == "__main__":
    main()
