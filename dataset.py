import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random

class CustomDataset(Dataset):
    def __init__(self, root_dir, phase='train', transform=None, augment=False):
        self.transform = transform
        self.augment = augment

        # Directories for domain A and domain B images
        self.dir_A = os.path.join(root_dir, f"{phase}_A")
        self.dir_B = os.path.join(root_dir, f"{phase}_B")

        # List of images in each directory
        self.image_files_A = sorted([
            f for f in os.listdir(self.dir_A)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.image_files_B = sorted([
            f for f in os.listdir(self.dir_B)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        # Ensure that both directories have the same number of images
        assert len(self.image_files_A) == len(self.image_files_B), \
            "The number of images in train_A and train_B must be the same"

    def __len__(self):
        return len(self.image_files_A)

    def __getitem__(self, idx):
        img_path_A = os.path.join(self.dir_A, self.image_files_A[idx])
        img_path_B = os.path.join(self.dir_B, self.image_files_B[idx])

        image_A = Image.open(img_path_A).convert('RGB')
        image_B = Image.open(img_path_B).convert('RGB')

        if self.augment:
            # Apply augmentations consistently to both images
            image_A, image_B = self.apply_augmentations(image_A, image_B)

        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)

        return image_A, image_B

    def apply_augmentations(self, img_A, img_B):
        # Random Horizontal Flip
        if random.random() > 0.5:
            img_A = F.hflip(img_A)
            img_B = F.hflip(img_B)

        # Add other augmentations here, ensuring they're applied identically
        # For example, slight brightness adjustment
        brightness_factor = random.uniform(0.9, 1.1)
        img_A = F.adjust_brightness(img_A, brightness_factor)
        img_B = F.adjust_brightness(img_B, brightness_factor)

        return img_A, img_B



def get_transforms(image_size, normalize=True, mean=None, std=None):
    transforms_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    if normalize:
        if mean is None or std is None:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        transforms_list.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(transforms_list)

