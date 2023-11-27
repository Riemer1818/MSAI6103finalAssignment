import os
import re
import shutil

def copy_and_rename_files(source_dir, destination_dir):
    # Ensure destination directory exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for file in os.listdir(source_dir):
        # Check if 'fake' is in the filename
        if 'fake' in file:
            # Full path of the source file
            src_file_path = os.path.join(source_dir, file)
            # Replace '_fake_B' with '_leftImg8bit' in the filename
            new_file_name = re.sub(r'_fake_B', '_leftImg8bit', file)
            # Full path of the destination file
            dest_file_path = os.path.join(destination_dir, new_file_name)
            # Copy and rename the file
            shutil.copy(src_file_path, dest_file_path)

# Example usage
source_directory = '/home/msai/lidu0002/MSAI6103finalAssignment/results_try/cityscapes_pix2pix_VGG/test_latest/images'
destination_directory = '/home/msai/lidu0002/MSAI6103finalAssignment/results_try/datas_VGG'
copy_and_rename_files(source_directory, destination_directory)