
import os
from PIL import Image


TARGET_NUM_SAMPLES = 4


def create_augmented_data(class_path):
    existing_images = [f for f in os.listdir(
        class_path) if f.endswith(".png") and not f.startswith("aug")]

    num_existing_images = len(existing_images)

    num_augmented_required = TARGET_NUM_SAMPLES-num_existing_images

    num_per_sample = num_augmented_required // num_existing_images
    rem = num_augmented_required % num_existing_images

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

            # Calculate the position to paste the rotated image
            paste_position = ((image.size[0] - augmented_image.width) //
                              2, (image.size[1] - augmented_image.height) // 2)

            # Paste the rotated image onto the new image
            augmented_base.paste(augmented_image, paste_position)
            save_path = os.path.join(class_path, f"aug-{i}-{image_path}")
            augmented_base.save(save_path)


if __name__ == "__main__":
    target_class_path = "/home/white/uni_workspace/ecm3401-dissertation/data/ADNI_POST_PROCESS_SLICE/SMC"
    create_augmented_data(target_class_path)
