import os

# Define folder paths
before_folder = 'before'
after_folder = 'after'

# Starting number for renaming
start_number = 2000

# Patterns to rename
patterns_to_rename = ["train_input_", "validation_input_", "train_edited_", "validation_edited_"]


def rename_files(folder, start_number):
    """Rename files in the folder according to specified patterns, starting from start_number."""
    current_number = start_number

    for filename in os.listdir(folder):
        # Check if the filename starts with any of the patterns
        if any(filename.startswith(pattern) for pattern in patterns_to_rename) and filename.endswith('.jpg'):
            # Get the file extension and new name
            file_extension = os.path.splitext(filename)[1]
            new_filename = f"{current_number}{file_extension}"

            # Rename the file
            old_path = os.path.join(folder, filename)
            new_path = os.path.join(folder, new_filename)
            os.rename(old_path, new_path)

            print(f"Renamed {filename} to {new_filename}")
            current_number += 1


# Rename files in both 'before' and 'after' folders
rename_files(before_folder, start_number)
rename_files(after_folder, start_number)
