import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import shutil
from tqdm import tqdm

# Paths to the input folders
before_folder = "original"
after_folder = "after"
dataset_folder = "datasets/dataset"

# Define dataset split ratios
train_ratio = 0.8
val_ratio = 0.2

def create_directory_structure():
    dirs = ['train_A', 'train_B', 'val_A', 'val_B']
    for dir in dirs:
        path = os.path.join(dataset_folder, dir)
        os.makedirs(path, exist_ok=True)
        # Clear existing files
        for file in os.listdir(path):
            os.remove(os.path.join(path, file))

def apply_augmentations(image):
    # Convert to PIL Image if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    augmented = image.copy()
    
    # Random horizontal flip (50% chance)
    if random.random() < 0.5:
        augmented = ImageOps.mirror(augmented)
    
    # Brightness adjustment (±10%)
    brightness_factor = random.uniform(0.9, 1.1)
    augmented = ImageEnhance.Brightness(augmented).enhance(brightness_factor)
    
    # Contrast adjustment (±10%)
    contrast_factor = random.uniform(0.9, 1.1)
    augmented = ImageEnhance.Contrast(augmented).enhance(contrast_factor)
    
    # Color jittering (±10%)
    color_factor = random.uniform(0.9, 1.1)
    augmented = ImageEnhance.Color(augmented).enhance(color_factor)
    
    # Slight translation (±5% of image size)
    width, height = augmented.size
    max_dx = int(0.05 * width)
    max_dy = int(0.05 * height)
    dx = random.randint(-max_dx, max_dx)
    dy = random.randint(-max_dy, max_dy)
    augmented = augmented.transform(augmented.size, Image.AFFINE, (1, 0, dx, 0, 1, dy))
    
    # Slight scaling (±5%)
    scale_factor = random.uniform(0.95, 1.05)
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    augmented = augmented.resize((new_width, new_height))
    augmented = augmented.resize((width, height))  # Resize back to original size
    
    # Sharpness adjustment (±20%)
    sharpness_factor = random.uniform(0.8, 1.2)
    augmented = ImageEnhance.Sharpness(augmented).enhance(sharpness_factor)
    
    # Gamma correction (0.9 to 1.1)
    gamma = random.uniform(0.9, 1.1)
    augmented = Image.fromarray(np.uint8(255 * (np.array(augmented) / 255) ** gamma))
    
    return augmented

def process_dataset():
    # Get all image files
    before_images = set(os.listdir(before_folder))
    after_images = set(os.listdir(after_folder))
    
    # Find common image names (matching pairs)
    common_images = sorted(before_images.intersection(after_images))
    
    # Report on unpaired images
    unpaired_before = before_images - after_images
    unpaired_after = after_images - before_images
    
    if unpaired_before:
        print(f"\nSkipping {len(unpaired_before)} unpaired images from 'original' folder:")
        for img in sorted(unpaired_before):
            print(f"- {img}")
    
    if unpaired_after:
        print(f"\nSkipping {len(unpaired_after)} unpaired images from 'after' folder:")
        for img in sorted(unpaired_after):
            print(f"- {img}")
    
    print(f"\nProcessing {len(common_images)} paired images...")
    
    # Create pairs of images with index to maintain order
    image_pairs = list(enumerate(common_images))
    random.shuffle(image_pairs)
    
    # Split into train and validation sets
    split_idx = int(len(image_pairs) * train_ratio)
    train_pairs = image_pairs[:split_idx]
    val_pairs = image_pairs[split_idx:]
    
    # Process training set with multiple augmentations per image
    print("\nProcessing training set...")
    for original_idx, img_name in tqdm(train_pairs):
        # Load original images
        before_path = os.path.join(before_folder, img_name)
        after_path = os.path.join(after_folder, img_name)
        
        img_before = Image.open(before_path)
        img_after = Image.open(after_path)
        
        # Create multiple augmented versions
        for aug_idx in range(3):  # Create 3 augmented versions per image
            # Apply same augmentation to both images
            seed = random.randint(0, 2**32 - 1)
            
            random.seed(seed)
            aug_before = apply_augmentations(img_before)
            
            random.seed(seed)
            aug_after = apply_augmentations(img_after)
            
            # Save augmented images with original index to maintain pairing
            pair_id = f"{original_idx:04d}"  # Zero-pad to 4 digits
            aug_before.save(os.path.join(dataset_folder, 'train_A', f'pair_{pair_id}_aug{aug_idx}.png'))
            aug_after.save(os.path.join(dataset_folder, 'train_B', f'pair_{pair_id}_aug{aug_idx}.png'))
    
    # Process validation set (no augmentation)
    print("\nProcessing validation set...")
    for original_idx, img_name in tqdm(val_pairs):
        before_path = os.path.join(before_folder, img_name)
        after_path = os.path.join(after_folder, img_name)
        
        # Copy original images without augmentation, using consistent naming
        pair_id = f"{original_idx:04d}"  # Zero-pad to 4 digits
        shutil.copy(before_path, os.path.join(dataset_folder, 'val_A', f'pair_{pair_id}.png'))
        shutil.copy(after_path, os.path.join(dataset_folder, 'val_B', f'pair_{pair_id}.png'))

    # Verify pairing
    def verify_pairs(dir_a, dir_b):
        files_a = sorted(os.listdir(dir_a))
        files_b = sorted(os.listdir(dir_b))
        if len(files_a) != len(files_b):
            print(f"Warning: Number of files in {dir_a} ({len(files_a)}) and {dir_b} ({len(files_b)}) don't match!")
            return False
        for fa, fb in zip(files_a, files_b):
            if fa.split('_aug')[0] != fb.split('_aug')[0]:
                print(f"Warning: Mismatched pair found: {fa} and {fb}")
                return False
        return True

    # Verify both training and validation pairs
    train_a_dir = os.path.join(dataset_folder, 'train_A')
    train_b_dir = os.path.join(dataset_folder, 'train_B')
    val_a_dir = os.path.join(dataset_folder, 'val_A')
    val_b_dir = os.path.join(dataset_folder, 'val_B')

    print("\nVerifying pairs...")
    if verify_pairs(train_a_dir, train_b_dir):
        print("Training pairs verified successfully!")
    if verify_pairs(val_a_dir, val_b_dir):
        print("Validation pairs verified successfully!")
    
    # Print final statistics
    print(f"\nDataset creation complete!")
    print(f"Total paired images processed: {len(common_images)}")
    print(f"Training pairs: {len(train_pairs)} (x3 with augmentation)")
    print(f"Validation pairs: {len(val_pairs)}")

if __name__ == "__main__":
    print("Creating directory structure...")
    create_directory_structure()
    print("Processing dataset...")
    process_dataset() 