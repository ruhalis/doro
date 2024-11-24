import os
import random
from PIL import Image, ImageEnhance, ImageOps

# Paths to the input folders
before_folder = "before"
after_folder = "after"
dataset_folder = "dataset"

# Define dataset split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Create directories for train, val, and test
train_input_folder = os.path.join(dataset_folder, "train", "input")
train_target_folder = os.path.join(dataset_folder, "train", "target")
val_input_folder = os.path.join(dataset_folder, "val", "input")
val_target_folder = os.path.join(dataset_folder, "val", "target")
test_input_folder = os.path.join(dataset_folder, "test", "input")
test_target_folder = os.path.join(dataset_folder, "test", "target")

os.makedirs(train_input_folder, exist_ok=True)
os.makedirs(train_target_folder, exist_ok=True)
os.makedirs(val_input_folder, exist_ok=True)
os.makedirs(val_target_folder, exist_ok=True)
os.makedirs(test_input_folder, exist_ok=True)
os.makedirs(test_target_folder, exist_ok=True)

# Get all the filenames from 'before' and match with 'after'
before_files = sorted(os.listdir(before_folder))
after_files = sorted(os.listdir(after_folder))

# Ensure filenames align
pairs = [(bf, af) for bf, af in zip(before_files, after_files) if os.path.splitext(bf)[0] == os.path.splitext(af)[0]]

# Shuffle the data and split into train, val, and test
random.shuffle(pairs)
total_pairs = len(pairs)
train_pairs = pairs[:int(total_pairs * train_ratio)]
val_pairs = pairs[int(total_pairs * train_ratio):int(total_pairs * (train_ratio + val_ratio))]
test_pairs = pairs[int(total_pairs * (train_ratio + val_ratio)):]

# Function to resize and save an image
def resize_and_save(src_path, dst_path, size=(256, 256)):
    img = Image.open(src_path).convert("RGB")  # Ensure RGB format
    img = img.resize(size, Image.ANTIALIAS)
    img.save(dst_path, "PNG")

# Function to apply random transformations
def apply_transformations(img_input, img_target):
    # Random horizontal flip
    if random.random() < 0.5:
        img_input = img_input.transpose(Image.FLIP_LEFT_RIGHT)
        img_target = img_target.transpose(Image.FLIP_LEFT_RIGHT)

    # Random rotation
    angle = random.randint(-15, 15)  # Rotate between -15 to 15 degrees
    img_input = img_input.rotate(angle, expand=True)
    img_target = img_target.rotate(angle, expand=True)

    # Shared random brightness adjustment
    brightness_factor = random.uniform(0.7, 1.3)
    enhancer = ImageEnhance.Brightness(img_input)
    img_input = enhancer.enhance(brightness_factor)
    enhancer = ImageEnhance.Brightness(img_target)
    img_target = enhancer.enhance(brightness_factor)

    # Shared random contrast adjustment
    contrast_factor = random.uniform(0.7, 1.3)
    enhancer = ImageEnhance.Contrast(img_input)
    img_input = enhancer.enhance(contrast_factor)
    enhancer = ImageEnhance.Contrast(img_target)
    img_target = enhancer.enhance(contrast_factor)

    # Shared random color adjustment
    color_factor = random.uniform(0.7, 1.3)
    enhancer = ImageEnhance.Color(img_input)
    img_input = enhancer.enhance(color_factor)
    enhancer = ImageEnhance.Color(img_target)
    img_target = enhancer.enhance(color_factor)

    # Simulate random lighting conditions (add shadow effect)
    if random.random() < 0.3:
        shadow = ImageOps.colorize(
            ImageOps.grayscale(img_input), black="black", white="gray"
        ).convert("RGBA")
        img_input = Image.blend(img_input.convert("RGBA"), shadow, alpha=0.3).convert("RGB")

        shadow = ImageOps.colorize(
            ImageOps.grayscale(img_target), black="black", white="gray"
        ).convert("RGBA")
        img_target = Image.blend(img_target.convert("RGBA"), shadow, alpha=0.3).convert("RGB")

    return img_input, img_target

# Function to process image pairs and save to destination
def process_pairs(pairs, input_folder, target_folder, start_index, augment_probability=0.5):
    index = start_index
    for bf, af in pairs:
        input_path = os.path.join(before_folder, bf)
        target_path = os.path.join(after_folder, af)

        # Save original images
        input_save_path = os.path.join(input_folder, f"{index}_input.png")
        target_save_path = os.path.join(target_folder, f"{index}_target.png")
        resize_and_save(input_path, input_save_path)
        resize_and_save(target_path, target_save_path)
        index += 1

        # Randomly apply transformations and save augmented images
        if random.random() < augment_probability:
            img_input = Image.open(input_path).convert("RGB")
            img_target = Image.open(target_path).convert("RGB")

            img_input_aug, img_target_aug = apply_transformations(img_input, img_target)

            input_aug_save_path = os.path.join(input_folder, f"{index}_input.png")
            target_aug_save_path = os.path.join(target_folder, f"{index}_target.png")
            img_input_aug.save(input_aug_save_path, "PNG")
            img_target_aug.save(target_aug_save_path, "PNG")
            index += 1

    return index

# Process each split
start_index = 1
start_index = process_pairs(train_pairs, train_input_folder, train_target_folder, start_index)
start_index = process_pairs(val_pairs, val_input_folder, val_target_folder, start_index)
process_pairs(test_pairs, test_input_folder, test_target_folder, start_index)

print("Dataset preparation with random pair transformations completed!")
