import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir, phase='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.phase = phase
        
        # Get image paths
        self.A_path = os.path.join(root_dir, f'{phase}_A')
        self.B_path = os.path.join(root_dir, f'{phase}_B')
        
        self.A_images = sorted(os.listdir(self.A_path))
        self.B_images = sorted(os.listdir(self.B_path))
        
        assert len(self.A_images) == len(self.B_images), "Unequal number of images in A and B"

    def __len__(self):
        return len(self.A_images)

    def __getitem__(self, idx):
        A_img = Image.open(os.path.join(self.A_path, self.A_images[idx])).convert('RGB')
        B_img = Image.open(os.path.join(self.B_path, self.B_images[idx])).convert('RGB')
        
        if self.transform:
            A_img = self.transform(A_img)
            B_img = self.transform(B_img)
            
        return A_img, B_img

def get_transforms(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), 
                        interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
