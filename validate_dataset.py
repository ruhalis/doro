import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
from torchvision import transforms
import shutil

def validate_dataset(dataset_path):
    """Validate the dataset structure and content"""
    print("\n=== Starting Dataset Validation ===\n")
    
    # Check directory structure
    required_dirs = ['train_A', 'train_B', 'val_A', 'val_B']
    for dir_name in required_dirs:
        dir_path = os.path.join(dataset_path, dir_name)
        if not os.path.exists(dir_path):
            print(f"❌ Error: Directory {dir_path} does not exist!")
            return False
    print("✅ Directory structure verified")

    # Validate pairs and image properties
    for split in ['train', 'val']:
        dir_A = os.path.join(dataset_path, f'{split}_A')
        dir_B = os.path.join(dataset_path, f'{split}_B')
        
        files_A = sorted(os.listdir(dir_A))
        files_B = sorted(os.listdir(dir_B))
        
        print(f"\nValidating {split} split:")
        print(f"Found {len(files_A)} images in {split}_A")
        print(f"Found {len(files_B)} images in {split}_B")
        
        if len(files_A) != len(files_B):
            print(f"❌ Error: Unequal number of images in {split}_A and {split}_B!")
            return False
        
        # Validate image pairs and properties
        corrupted_images = []
        mismatched_sizes = []
        
        for img_A, img_B in tqdm(zip(files_A, files_B), total=len(files_A), desc=f"Checking {split} images"):
            # Verify pair naming
            base_A = img_A.split('_aug')[0]
            base_B = img_B.split('_aug')[0]
            
            if base_A != base_B:
                print(f"❌ Error: Mismatched pair found: {img_A} and {img_B}")
                return False
            
            # Check if images can be opened and have valid content
            try:
                path_A = os.path.join(dir_A, img_A)
                path_B = os.path.join(dir_B, img_B)
                
                img_data_A = Image.open(path_A)
                img_data_B = Image.open(path_B)
                
                # Check image sizes
                if img_data_A.size != img_data_B.size:
                    mismatched_sizes.append((img_A, img_B))
                
                # Check if images are not corrupted
                img_data_A.verify()
                img_data_B.verify()
                
                # Check if images have valid content (not empty or too small)
                min_size = 64
                if img_data_A.size[0] < min_size or img_data_A.size[1] < min_size:
                    corrupted_images.append(img_A)
                if img_data_B.size[0] < min_size or img_data_B.size[1] < min_size:
                    corrupted_images.append(img_B)
                
            except Exception as e:
                print(f"❌ Error processing images {img_A} or {img_B}: {str(e)}")
                corrupted_images.extend([img_A, img_B])
        
        # Report issues
        if corrupted_images:
            print(f"\n❌ Found {len(corrupted_images)} corrupted or invalid images:")
            for img in corrupted_images:
                print(f"  - {img}")
        
        if mismatched_sizes:
            print(f"\n❌ Found {len(mismatched_sizes)} pairs with mismatched sizes:")
            for img_A, img_B in mismatched_sizes:
                print(f"  - {img_A} and {img_B}")
        
        if not corrupted_images and not mismatched_sizes:
            print(f"✅ All {split} images validated successfully")
    
    # Validate augmentations in training set
    train_files = sorted(os.listdir(os.path.join(dataset_path, 'train_A')))
    aug_counts = {}
    for file in train_files:
        base_name = file.split('_aug')[0]
        aug_counts[base_name] = aug_counts.get(base_name, 0) + 1
    
    expected_augs = 3  # Expected number of augmentations per image
    incorrect_augs = {k: v for k, v in aug_counts.items() if v != expected_augs}
    
    if incorrect_augs:
        print(f"\n❌ Found {len(incorrect_augs)} images with incorrect number of augmentations:")
        for base_name, count in incorrect_augs.items():
            print(f"  - {base_name}: {count} augmentations (expected {expected_augs})")
    else:
        print(f"\n✅ All training images have correct number of augmentations ({expected_augs})")
    
    print("\n=== Dataset Validation Complete ===")
    return not (corrupted_images or mismatched_sizes or incorrect_augs)

def backup_dataset(dataset_path):
    """Create a backup of the dataset"""
    backup_path = f"{dataset_path}_backup"
    if os.path.exists(backup_path):
        shutil.rmtree(backup_path)
    shutil.copytree(dataset_path, backup_path)
    print(f"\n✅ Dataset backed up to {backup_path}")

if __name__ == "__main__":
    dataset_path = "datasets/dataset"  # Update this path if needed
    
    # Create backup before validation
    print("Creating dataset backup...")
    backup_dataset(dataset_path)
    
    # Validate dataset
    is_valid = validate_dataset(dataset_path)
    
    if is_valid:
        print("\n✅ Dataset validation passed successfully!")
    else:
        print("\n❌ Dataset validation failed! Please check the errors above.") 