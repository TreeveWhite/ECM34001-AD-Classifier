
import os
import sys
from PIL import Image

DATASET_BASE_PATH = "/home/white/uni_workspace/ecm3401-dissertation/data/ADNI_SLICE_TRAINING_DATA"

CLASSES = ["GOOD", "BAD"]


def create_augmented_data(class_path, target_num_images):
    existing_images = [f for f in os.listdir(
        class_path) if f.endswith(".png") and not f.startswith("aug")]

    num_existing_images = len(existing_images)

    num_augmented_required = target_num_images-num_existing_images

    num_per_sample = num_augmented_required // num_existing_images
    rem = num_augmented_required % num_existing_images

    if num_per_sample < 0:
        print(f"Already have enough images in {class_path}")

    for image_path in existing_images:
        todo = num_per_sample
        if rem != 0:
            rem -= 1
            todo += 1
        image = Image.open(os.path.join(class_path, image_path))
        for i in range(todo):
            augmented_image = image.rotate(360/(num_per_sample+1) * i)
            if i % 2 == 0:
                augmented_image = augmented_image.transpose(
                    Image.FLIP_TOP_BOTTOM)

            augmented_base = Image.new("RGB", image.size, "black")

            paste_position = ((image.size[0] - augmented_image.width) //
                              2, (image.size[1] - augmented_image.height) // 2)

            augmented_base.paste(augmented_image, paste_position)
            save_path = os.path.join(class_path, f"aug-{i}-{image_path}")
            augmented_base.save(save_path)


if __name__ == "__main__":

    args = sys.argv[1:]

    target_classes = [c for c in args if c in CLASSES]

    desired_num_images = int(args[-1])

    for target_class in target_classes:
        create_augmented_data(os.path.join(
            DATASET_BASE_PATH, target_class), desired_num_images)
