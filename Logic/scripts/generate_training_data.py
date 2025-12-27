import json
import os
import math
import random
import argparse
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from tqdm import tqdm  # New Import

# --- CONFIGURATION ---
FONT_SIZE_RANGE = (12, 48)
JITTER_MAX = 2 
BLUR_PROBABILITY = 0.30 

def load_font_list(path):
    if not os.path.exists(path):
        return []
    fonts = []
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                clean = line.strip()
                if clean and not clean.startswith('#'):
                    if not clean.lower().endswith('.ttf'):
                        clean += '.ttf'
                    fonts.append(clean)
    except: pass
    return fonts

def synthesize_image(text, font_path, rotation, size, save_path, blur_radius=0):
    try:
        font = ImageFont.truetype(font_path, size)
        temp_draw = ImageDraw.Draw(Image.new('L', (1, 1)))
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        rad = math.radians(abs(rotation))
        nw = int(tw * math.cos(rad) + th * math.sin(rad)) + 40
        nh = int(tw * math.sin(rad) + th * math.cos(rad)) + 40

        img = Image.new('L', (nw, nh), color=0)
        draw = ImageDraw.Draw(img)
        cx = (nw - tw) // 2 - bbox[0]
        cy = (nh - th) // 2 - bbox[1]

        draw.text((cx + random.randint(-JITTER_MAX, JITTER_MAX), 
                   cy + random.randint(-JITTER_MAX, JITTER_MAX)), 
                  text, font=font, fill=255)
        
        if rotation != 0:
            img = img.rotate(rotation, resample=Image.BICUBIC, expand=True)
        if blur_radius > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        content_bbox = img.getbbox()
        if content_bbox:
            m = 5
            img = img.crop((content_bbox[0]-m, content_bbox[1]-m, 
                            content_bbox[2]+m, content_bbox[3]+m))
        img.save(save_path)
        return img.size 
    except:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Text or path to .txt")
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--fonts", action="store_true")
    parser.add_argument("--output", help="Custom output directory")
    args = parser.parse_args()

    # --- PATHS ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
    DATA_ROOT = os.path.join(ROOT_DIR, 'Data')
    
    OUT_BASE = os.path.abspath(args.output) if args.output else DATA_ROOT
    IMG_DIR = os.path.join(OUT_BASE, 'document training data', 'data')
    MANIFEST_DIR = os.path.join(OUT_BASE, 'document training data', 'directory')
    MANIFEST_PATH = os.path.join(MANIFEST_DIR, 'dataset_manifest.jsonl')

    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(MANIFEST_DIR, exist_ok=True)

    # 1. Load Fonts
    fonts_to_use = load_font_list(os.path.join(DATA_ROOT, 'lists', 'list-of-fonts.txt')) if args.fonts else ["arial.ttf"]
    if not fonts_to_use: fonts_to_use = ["arial.ttf"]

    # 2. Load Lines
    all_lines = []
    if os.path.isfile(args.input):
        with open(args.input, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                content = line.strip()
                if content and not content.startswith('---'):
                    all_lines.append(content)
    else:
        all_lines = [args.input]

    rotations = list(range(0, 360, 15)) if args.rotate else [0]
    
    # 3. Counter Resumption
    global_counter = 0
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
            global_counter = sum(1 for _ in f)

    # Calculate total iterations for progress bar
    total_to_generate = len(all_lines) * len(fonts_to_use) * len(rotations)

    print(f"--- Starting Generation ---")
    print(f"Total source lines: {len(all_lines)}")
    print(f"Total images to create: {total_to_generate}")

    # 4. Loop with tqdm
    with open(MANIFEST_PATH, 'a', encoding='utf-8') as f_out:
        with tqdm(total=total_to_generate, desc="Generating Images", unit="img") as pbar:
            for text in all_lines:
                # Update description to show current text snippet
                pbar.set_postfix_str(f"Text: {text[:15]}...")
                
                for f_name in fonts_to_use:
                    f_path = os.path.join(DATA_ROOT, 'fonts', f_name)
                    if not os.path.exists(f_path):
                        f_path = os.path.join("C:\\Windows\\Fonts", f_name)

                    for deg in rotations:
                        current_size = random.randint(FONT_SIZE_RANGE[0], FONT_SIZE_RANGE[1])
                        img_id = f"{global_counter:08d}"
                        file_name = f"{img_id}.png"
                        save_path = os.path.join(IMG_DIR, file_name)
                        
                        blur = random.uniform(0.5, 1.3) if random.random() < BLUR_PROBABILITY else 0
                        dims = synthesize_image(text, f_path, deg, current_size, save_path, blur)
                        
                        if dims:
                            entry = {
                                "id": img_id, 
                                "filename": file_name,
                                "text": text, 
                                "font": f_name,
                                "font_size": current_size,
                                "rotation": deg, 
                                "width": dims[0], 
                                "height": dims[1]
                            }
                            f_out.write(json.dumps(entry) + "\n")
                            global_counter += 1
                        
                        pbar.update(1) # Advance the progress bar by 1

    print(f"\nFinished. Total images in dataset: {global_counter}")

if __name__ == "__main__":
    main()