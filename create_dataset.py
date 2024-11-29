import os
import random
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm

# Paths to the input folders
before_folder = "original"
after_folder = "after"
dataset_folder = "datasets/dataset"

# Define dataset split ratios
train_ratio = 0.9  # Increased train ratio for synthetic data
val_ratio = 0.1    # Reduced validation ratio

def create_directory_structure():
    dirs = ['train_A', 'train_B', 'val_A', 'val_B']
    for dir in dirs:
        path = os.path.join(dataset_folder, dir)
        os.makedirs(path, exist_ok=True)
        for file in os.listdir(path):
            os.remove(os.path.join(path, file))

def process_dataset():
    # Get all image files
    before_images = set(os.listdir(before_folder))
    after_images = set(os.listdir(after_folder))
    common_images = sorted(before_images.intersection(after_images))
    
    # Create pairs and split dataset
    image_pairs = list(enumerate(common_images))
    random.shuffle(image_pairs)
    
    split_idx = int(len(image_pairs) * train_ratio)
    train_pairs = image_pairs[:split_idx]
    val_pairs = image_pairs[split_idx:]
    
    # Process training set (minimal augmentation for synthetic data)
    print("\nProcessing training set...")
    for original_idx, img_name in tqdm(train_pairs):
        before_path = os.path.join(before_folder, img_name)
        after_path = os.path.join(after_folder, img_name)
        
        # Load images in high quality
        img_before = Image.open(before_path).convert('RGB')
        img_after = Image.open(after_path).convert('RGB')
        
        # Save original pair
        pair_id = f"{original_idx:04d}"
        img_before.save(
            os.path.join(dataset_folder, 'train_A', f'pair_{pair_id}.png'),
            format='PNG',
            optimize=False  # Preserve exact pixel values
        )
        img_after.save(
            os.path.join(dataset_folder, 'train_B', f'pair_{pair_id}.png'),
            format='PNG',
            optimize=False
        )
        
        # Only add horizontal flip for synthetic data
        if random.random() < 0.5:
            img_before_flip = img_before.transpose(Image.FLIP_LEFT_RIGHT)
            img_after_flip = img_after.transpose(Image.FLIP_LEFT_RIGHT)
            
            img_before_flip.save(
                os.path.join(dataset_folder, 'train_A', f'pair_{pair_id}_flip.png'),
                format='PNG',
                optimize=False
            )
            img_after_flip.save(
                os.path.join(dataset_folder, 'train_B', f'pair_{pair_id}_flip.png'),
                format='PNG',
                optimize=False
            )
    
    # Process validation set (no augmentation)
    print("\nProcessing validation set...")
    for original_idx, img_name in tqdm(val_pairs):
        before_path = os.path.join(before_folder, img_name)
        after_path = os.path.join(after_folder, img_name)
        
        pair_id = f"{original_idx:04d}"
        shutil.copy(before_path, os.path.join(dataset_folder, 'val_A', f'pair_{pair_id}.png'))
        shutil.copy(after_path, os.path.join(dataset_folder, 'val_B', f'pair_{pair_id}.png'))

if __name__ == "__main__":
    print("Creating directory structure...")
    create_directory_structure()
    print("Processing dataset...")
    process_dataset() 