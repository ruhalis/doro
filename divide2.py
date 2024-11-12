import os
from PIL import Image


def split_images_in_folder(input_folder, output_left_folder, output_right_folder):
    # Create output folders if they don't exist
    os.makedirs(output_left_folder, exist_ok=True)
    os.makedirs(output_right_folder, exist_ok=True)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            # Construct full file path
            file_path = os.path.join(input_folder, filename)

            # Open the image
            image = Image.open(file_path)
            width, height = image.size
            middle = width // 2

            # Split the image
            left_image = image.crop((0, 0, middle, height))
            right_image = image.crop((middle, 0, width, height))

            # Define output paths with original filename
            left_image_path = os.path.join(output_left_folder, filename.replace('.webp', '.png'))
            right_image_path = os.path.join(output_right_folder, filename.replace('.webp', '.png'))

            # Save images as PNG, preserving the original name
            left_image.save(left_image_path, "PNG")
            right_image.save(right_image_path, "PNG")

            print(f"Processed {filename}")


# Usage example
split_images_in_folder(
    input_folder="full",
    output_left_folder="output_left_folder",
    output_right_folder="output_right_folder"
)
