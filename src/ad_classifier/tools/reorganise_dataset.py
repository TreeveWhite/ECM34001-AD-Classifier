import os
import random
import shutil

source_dir = "/mnt/c/Users/treev/Desktop/ADNI_POST_PROCESS_MODELED_SLICE_AUG"

destination_dir = "/mnt/c/Users/treev/Desktop/ADNI_TEST_DATASET"

test_percentage = 0.3

if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

for class_dir in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_dir)
    if os.path.isdir(class_path):
        files = os.listdir(class_path)
        num_files_to_move = int(len(files) * test_percentage)

        files_to_move = random.sample(files, num_files_to_move)

        for file_name in files_to_move:
            source_file = os.path.join(class_path, file_name)
            destination_file = os.path.join(
                destination_dir, class_dir, file_name)

            if not os.path.exists(os.path.join(destination_dir, class_dir)):
                os.makedirs(os.path.join(destination_dir, class_dir))
            shutil.move(source_file, destination_file)
            print(f"Moved {file_name} to {destination_file}")
