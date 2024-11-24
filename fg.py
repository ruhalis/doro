from PIL import Image
import os
import glob

# Input directory (the folder containing your images)
input_dir = '/images'  # Replace with your actual folder path


# Resize images in place and save them as JPG
def resize_and_replace_images_as_jpg(input_dir, size=(1024, 1024)):
    for img_path in glob.glob(os.path.join(input_dir, '*.*')):  # Loop through all files in input_dir
        try:
            with Image.open(img_path) as img:
                # Convert image to RGB mode if not already in RGB
                img = img.convert("RGB")

                # Resize image while maintaining aspect ratio
                img.thumbnail(size, Image.ANTIALIAS)

                # Create a blank canvas (black background) to center the resized image
                new_img = Image.new("RGB", size, (0, 0, 0))  # Black background
                new_img.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))

                # Save the resized image as JPG, replacing the original
                base_name = os.path.splitext(os.path.basename(img_path))[0]  # Get filename without extension
                new_img.save(os.path.join(input_dir, f"{base_name}.jpg"), "JPEG", quality=95)  # Save as JPG
                print(f"Resized and saved as JPG: {base_name}.jpg")

                # Optionally, delete the original file if it was not a JPG
                if not img_path.lower().endswith(".jpg"):
                    os.remove(img_path)
                    print(f"Deleted original file: {img_path}")
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")


# Call the function
resize_and_replace_images_as_jpg(input_dir)
