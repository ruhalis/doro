import os
import shutil
import re

# Define your folder paths
train_input_folder = 'D:/doro/train_input'
validation_input_folder = 'D:/doro/validation_input'
train_edited_folder = 'D:/doro/train_edited'
validation_edited_folder = 'D:/doro/validation_edited'


# Define output folders if you need to copy matched files
output_folder = 'D:/doro/output'
os.makedirs(output_folder, exist_ok=True)


def get_image_number(filename, prefix):
    """Extract the numeric suffix from a filename with a specified prefix."""
    match = re.search(f"{prefix}(\\d+)\\.jpg$", filename)
    return match.group(1) if match else None


def match_and_copy_images(input_folder, edited_folder, prefix_input, prefix_edited):
    """Match images in input and edited folders by number suffix and copy them to an output folder."""
    input_images = [img for img in os.listdir(input_folder) if img.startswith(prefix_input) and img.endswith('.jpg')]
    edited_images = [img for img in os.listdir(edited_folder) if img.startswith(prefix_edited) and img.endswith('.jpg')]

    matched_images = []
    for input_image in input_images:
        input_number = get_image_number(input_image, prefix_input)

        if input_number is not None:
            edited_image = f"{prefix_edited}{input_number}.jpg"

            if edited_image in edited_images:
                input_path = os.path.join(input_folder, input_image)
                edited_path = os.path.join(edited_folder, edited_image)
                matched_images.append((input_path, edited_path))

                # Copy the matched files to the output folder, if needed
                shutil.copy(input_path, os.path.join(output_folder, input_image))
                shutil.copy(edited_path, os.path.join(output_folder, edited_image))

    return matched_images


# Run matching for both train and validation folders
train_matches = match_and_copy_images(train_input_folder, train_edited_folder, 'train_input_', 'train_edited_')
validation_matches = match_and_copy_images(validation_input_folder, validation_edited_folder, 'validation_input_',
                                           'validation_edited_')

# Print the matches for verification
print("Train matches:", train_matches)
print("Validation matches:", validation_matches)