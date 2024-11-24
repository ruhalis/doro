import os
import shutil
from pathlib import Path


def create_cyclegan_dataset(before_dir, after_dir, output_dir):
    """
    Organizes paired images into CycleGAN's unpaired dataset structure.

    Parameters:
        before_dir (str): Path to the folder containing 'before' images.
        after_dir (str): Path to the folder containing 'after' images.
        output_dir (str): Path to the output folder for the CycleGAN dataset.
    """
    # Create output directories
    trainA_dir = os.path.join(output_dir, 'trainA')
    trainB_dir = os.path.join(output_dir, 'trainB')
    os.makedirs(trainA_dir, exist_ok=True)
    os.makedirs(trainB_dir, exist_ok=True)

    # List all files in the 'before' and 'after' directories
    before_files = sorted(Path(before_dir).glob('*'))
    after_files = sorted(Path(after_dir).glob('*'))

    # Ensure file extension matching for pair consistency
    before_files = [f for f in before_files if f.suffix in ['.jpg', '.jpeg', '.png']]
    after_files = [f for f in after_files if f.suffix in ['.jpg', '.jpeg', '.png']]

    # Track unmatched files
    unmatched_after_files = set(after_files)

    # Match files by their names
    matched_pairs = []
    for before_file in before_files:
        before_name = before_file.stem  # Get the name without the extension
        matched_file = None
        for after_file in unmatched_after_files:
            after_name = after_file.stem  # Get the name without the extension
            if before_name in after_name or after_name in before_name:
                matched_file = after_file
                break
        if matched_file:
            matched_pairs.append((before_file, matched_file))
            unmatched_after_files.remove(matched_file)

    # Copy matched files to CycleGAN dataset structure
    for i, (before_file, after_file) in enumerate(matched_pairs):
        # Define new filenames
        new_name_A = f'{i:05d}.jpg'
        new_name_B = f'{i:05d}.jpg'

        # Copy to the respective directories
        shutil.copy(before_file, os.path.join(trainA_dir, new_name_A))
        shutil.copy(after_file, os.path.join(trainB_dir, new_name_B))

    # Summary
    print(f"Dataset created successfully in '{output_dir}'")
    print(f"Total paired images processed: {len(matched_pairs)}")
    print(f"Unmatched 'after' images: {len(unmatched_after_files)}")

# Define input and output directories
before_dir = 'before'
after_dir = 'after'
output_dir = 'dataset'
# Call the function to create the dataset
create_cyclegan_dataset(before_dir, after_dir, output_dir)
