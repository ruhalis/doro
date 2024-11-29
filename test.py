import torch
from torchvision import transforms
from PIL import Image
import os
from models.attgan import Generator
import argparse
from tqdm import tqdm
import gc  # for garbage collection

class Tester:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        print(f"Using device: {device}")
        
        # Check CUDA availability
        if device == 'cuda':
            if torch.cuda.is_available():
                print(f"GPU: {torch.cuda.get_device_name()}")
                print(f"Memory Allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
            else:
                print("CUDA is not available. Falling back to CPU")
                self.device = 'cpu'
        
        # Initialize generator
        self.G = Generator().to(self.device)
        
        # Load checkpoint with error handling
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'G_state_dict' in checkpoint:
                self.G.load_state_dict(checkpoint['G_state_dict'])
            else:
                self.G.load_state_dict(checkpoint)
            print("Model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {str(e)}")
        
        self.G.eval()
        
        # Setup transforms for 512x512 images
        self.transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.inverse_transform = transforms.Compose([
            transforms.Normalize((-1, -1, -1), (2, 2, 2)),
            transforms.ToPILImage()
        ])

    @torch.no_grad()
    def process_image(self, image_path, save_path):
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Process through generator
            with torch.cuda.amp.autocast():  # Use mixed precision for better performance
                fake_img = self.G(img_tensor)
            
            # Move to CPU and convert to PIL
            fake_img = fake_img.cpu()
            fake_img = self.inverse_transform(fake_img.squeeze(0))
            
            # Save the result
            fake_img.save(save_path)
            
            # Clear GPU memory
            self.clear_gpu_memory()
            
            return True
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            self.clear_gpu_memory()
            return False

    def clear_gpu_memory(self):
        """Clear GPU memory cache"""
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

def main():
    parser = argparse.ArgumentParser(description='Test AttGAN on images')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output', type=str, required=True, help='Path to output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing')
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
        # Directory of images
        image_files = [f for f in os.listdir(args.input) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Found {len(image_files)} images to process")
        
        for img_file in tqdm(image_files):
            input_path = os.path.join(args.input, img_file)
            output_path = os.path.join(args.output, 'result_' + img_file)
            success = tester.process_image(input_path, output_path)
            if not success:
                print(f"Failed to process {img_file}")

if __name__ == "__main__":
    main()
