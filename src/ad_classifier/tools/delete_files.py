import os


def delete_files(directory):
    files = os.listdir(directory)
    files.sort()
    count = 0

    for file in files:
        count += 1
        if count % 5 != 1:
            file_path = os.path.join(directory, file)
            os.remove(file_path)
            print(f"Deleted: {file_path}")


if __name__ == "__main__":
    directory_path = "/home/white/uni_workspace/ecm3401-dissertation/data/ADNI_SLICE_TRAINING_DATA/BAD"
    delete_files(directory_path)
