import os
import sys
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def process_image(file_path, src_dir, out_dir):
    try:
        # Create identical folder structure
        rel_path = os.path.relpath(os.path.dirname(file_path), src_dir)
        new_dir = os.path.join(out_dir, rel_path)
        os.makedirs(new_dir, exist_ok=True)

        # Process the image
        img = Image.open(file_path).convert("RGB")

        # Upscale if needed
        if img.width < 224 or img.height < 224:
            scale = max(224 / img.width, 224 / img.height)
            new_width = int(img.width * scale)
            new_height = int(img.height * scale)
            img = img.resize((new_width, new_height), Image.LANCZOS)

        # Downscale if needed (maintaining aspect ratio)
        if img.width > 256 or img.height > 256:
            if img.width > img.height:
                new_width = 256
                new_height = int(256 * img.height / img.width)
            else:
                new_height = 256
                new_width = int(256 * img.width / img.height)
            img = img.resize((new_width, new_height), Image.LANCZOS)

        # Save optimized image
        out_path = os.path.join(new_dir, os.path.basename(file_path))
        img.save(out_path, "JPEG", quality=90, optimize=True)
        return 1
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0


def main():
    src_dir = "/dev/shm/AA2/Animals_with_Attributes2/JPEGImages"
    out_dir = "/dev/shm/AA2/Animals_with_Attributes2/JPEGImages_optimized"

    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # Collect all image files
    image_files = []
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg")):
                image_files.append(os.path.join(root, file))

    # Process images in parallel
    total_processed = 0
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(process_image, f, src_dir, out_dir) for f in image_files
        ]
        for future in tqdm(futures, total=len(futures)):
            total_processed += future.result()

    print(f"Successfully processed {total_processed} out of {len(image_files)} images")


if __name__ == "__main__":
    main()
