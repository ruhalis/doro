import os
import random
from PIL import Image, ImageEnhance, ImageOps

# Paths to the input folders
before_folder = "before"
after_folder = "after"
dataset_folder = "unpaired"

# Define dataset split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Create directories for train, val, and test for both domains
domains = ['trainA', 'trainB', 'valA', 'valB', 'testA', 'testB']
for domain in domains:
    for subfolder in ['input', 'target']:
        os.makedirs(os.path.join(dataset_folder, domain, subfolder), exist_ok=True)

# Get all the filenames from 'before' and 'after'
before_files = sorted([f for f in os.listdir(before_folder) if os.path.isfile(os.path.join(before_folder, f))])
after_files = sorted([f for f in os.listdir(after_folder) if os.path.isfile(os.path.join(after_folder, f))])

# Shuffle the data
random.shuffle(before_files)
random.shuffle(after_files)

# Calculate split indices
total_before = len(before_files)
total_after = len(after_files)

train_before = before_files[:int(total_before * train_ratio)]
val_before = before_files[int(total_before * train_ratio):int(total_before * (train_ratio + val_ratio))]
test_before = before_files[int(total_before * (train_ratio + val_ratio)):]

train_after = after_files[:int(total_after * train_ratio)]
val_after = after_files[int(total_after * train_ratio):int(total_after * (train_ratio + val_ratio))]
test_after = after_files[int(total_after * (train_ratio + val_ratio)):]

# Function to resize and save an image
def resize_and_save(src_path, dst_path, size=(256, 256)):
    img = Image.open(src_path).convert("RGB")  # Ensure RGB format
    img = img.resize(size, Image.ANTIALIAS)
    img.save(dst_path, "PNG")

# Function to apply random transformations
def apply_transformations(img):
    # Random horizontal flip
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Random rotation
    angle = random.randint(-15, 15)  # Rotate between -15 to 15 degrees
    img = img.rotate(angle, expand=True)

    # Shared random brightness adjustment
    brightness_factor = random.uniform(0.7, 1.3)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)

    # Shared random contrast adjustment
    contrast_factor = random.uniform(0.7, 1.3)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)

    # Shared random color adjustment
    color_factor = random.uniform(0.7, 1.3)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(color_factor)

    # Simulate random lighting conditions (add shadow effect)
    if random.random() < 0.3:
        shadow = ImageOps.colorize(
            ImageOps.grayscale(img), black="black", white="gray"
        ).convert("RGBA")
        img = Image.blend(img.convert("RGBA"), shadow, alpha=0.3).convert("RGB")

    return img

# Function to process and save images for a given domain
def process_domain(files, source_folder, target_subfolder, augment_probability=0.5):
    target_folder = os.path.join(dataset_folder, target_subfolder)
    index = 1  # Initialize index for naming

    for filename in files:
        src_path = os.path.join(source_folder, filename)
        # Define destination path with unique naming
        dst_path = os.path.join(target_folder, f"{index}_{target_subfolder}.png")
        resize_and_save(src_path, dst_path)
        index += 1

        # Apply random augmentations with a certain probability
        if random.random() < augment_probability:
            img = Image.open(src_path).convert("RGB")
            img_aug = apply_transformations(img)
            # Define augmented destination path
            dst_aug_path = os.path.join(target_folder, f"{index}_{target_subfolder}.png")
            resize_and_save(img_aug, dst_aug_path)
            index += 1

    return index

# Process each split for Domain A (Before Botox) and Domain B (After Botox)
# Domain A - Before Botox
trainA_files = train_before
valA_files = val_before
testA_files = test_before

# Domain B - After Botox
trainB_files = train_after
valB_files = val_after
testB_files = test_after

# Process Domain A
print("Processing Domain A (Before Botox)...")
start_index_A = process_domain(trainA_files, before_folder, "trainA/input")
process_domain(valA_files, before_folder, "valA/input")
process_domain(testA_files, before_folder, "testA/input")

# Process Domain B
print("Processing Domain B (After Botox)...")
start_index_B = process_domain(trainB_files, after_folder, "trainB/target")
process_domain(valB_files, after_folder, "valB/target")
process_domain(testB_files, after_folder, "testB/target")

print("Unpaired dataset preparation with random transformations completed!")
