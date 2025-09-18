import os
from PIL import Image

INPUT_TXT = "/data1/structure-clean/mmsegmentation-main/test_data/file_list_test_d_new.txt"
OUTPUT_ROOT = "/data1/structure-clean/mmsegmentation-main/test_data/test_images/"
OUTPUT_TXT = "/data1/structure-clean/mmsegmentation-main/test_data/test_images/test_resize_data.txt"

TARGET_WIDTH = 960
TARGET_HEIGHT = 544

def resize_and_pad(image, target_size, fill_color=(0, 0, 0), is_mask=False):
    target_w, target_h = target_size
    img_w, img_h = image.size

    ratio = min(target_w / img_w, target_h / img_h)
    new_w = int(img_w * ratio)
    new_h = int(img_h * ratio)
    
    resample = Image.NEAREST if is_mask else Image.LANCZOS
    
    resized_img = image.resize((new_w, new_h), resample)

    mode = "L" if is_mask else "RGB"
    new_img = Image.new(mode, (target_w, target_h), fill_color)

    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    new_img.paste(resized_img, (paste_x, paste_y))
    
    return new_img

def get_output_subdir(img_path):
    dir_path = os.path.dirname(img_path)
    path_components = [comp for comp in dir_path.split(os.sep) if comp]
    
    if len(path_components) >= 2:
        return os.path.join(path_components[-2], path_components[-1])
    elif path_components:
        return path_components[-1]
    return ""

def process_images():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    images_output_dir = os.path.join(OUTPUT_ROOT, "images")
    masks_output_dir = os.path.join(OUTPUT_ROOT, "masks")
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(masks_output_dir, exist_ok=True)
    
    try:
        with open(INPUT_TXT, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Error: File {INPUT_TXT} Not Exist!")
        return
    except Exception as e:
        print(f"Can't Read File: {str(e)}")
        return
    
    print(f"Found {len(lines)} image-mask pairs to process")
    
    try:
        output_file = open(OUTPUT_TXT, 'w', encoding='utf-8')
    except Exception as e:
        print(f"Can't create {OUTPUT_TXT}: {str(e)}")
        return
    
    processed_count = 0
    for idx, line in enumerate(lines):
        try:
            parts = line.split()
            if len(parts) < 2:
                print(f"Line {idx+1}: Skipped - expected 2 columns, found {len(parts)}")
                continue
                
            img_path = parts[0].strip()
            mask_path = parts[1].strip()
            
            with Image.open(img_path) as img:
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                
                img_subdir = get_output_subdir(img_path)
                img_output_dir = os.path.join(images_output_dir, img_subdir)
                os.makedirs(img_output_dir, exist_ok=True)
                
                processed_img = resize_and_pad(img, (TARGET_WIDTH, TARGET_HEIGHT))
                img_filename = os.path.basename(img_path)
                img_output_path = os.path.join(img_output_dir, img_filename)
                processed_img.save(img_output_path)
                
                with Image.open(mask_path) as mask:
                    if mask.mode != 'L':
                        mask = mask.convert('L')

                    mask_subdir = get_output_subdir(mask_path)
                    mask_output_dir = os.path.join(masks_output_dir, mask_subdir)
                    os.makedirs(mask_output_dir, exist_ok=True)

                    processed_mask = resize_and_pad(mask, (TARGET_WIDTH, TARGET_HEIGHT), fill_color=0, is_mask=True)
                    mask_filename = os.path.basename(mask_path)
                    mask_output_path = os.path.join(mask_output_dir, mask_filename)
                    processed_mask.save(mask_output_path)

                    abs_img_path = os.path.abspath(img_output_path)
                    abs_mask_path = os.path.abspath(mask_output_path)
                    output_file.write(f"{abs_img_path} {abs_mask_path}\n")
                    
                    print(f"Processed pair {idx+1}:")
                    print(f"  Image: {img_path} -> {img_output_path}")
                    print(f"  Mask: {mask_path} -> {mask_output_path}")
                    processed_count += 1
                    
        except Exception as e:
            print(f"Processing line {idx+1} failed: {str(e)}")
    
    output_file.close()
    print(f"\nProcessed {processed_count}/{len(lines)} image-mask pairs")
    print(f"Output saved to: {OUTPUT_TXT}")
    print(f"Images directory: {images_output_dir}")
    print(f"Masks directory: {masks_output_dir}")

if __name__ == "__main__":
    process_images()
