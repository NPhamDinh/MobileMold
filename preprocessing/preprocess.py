import os
import pandas as pd
from PIL import Image
from tqdm import tqdm  # for progress bar

# Configuration
METADATA_PATH = 'metadata.csv'
IMAGE_FOLDER = 'original'
OUTPUT_FOLDER = 'cropped_resized'
TARGET_SIZE = (224, 224)

# Crop parameters for each condition
CROP_PARAMS = {
    ('Mikroskop Phonesope 30x', 'Pixel 8 Pro'): {
        'left': 470, 'right': 300, 'top': 750, 'bottom': 950
    },
    ('Mikroskop Phonesope 30x', 'Galaxy S8+'): {
        'left': 430, 'right': 550, 'top': 1030, 'bottom': 932
    },
    ('Jiusion 30X', None): {  # None means phone doesn't matter for this microscope
        'left': 430, 'right': 600, 'top': 1040, 'bottom': 910
    }
}

def process_images():
    # Create output folder structure
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Read metadata if exists
    metadata = {}
    if os.path.exists(METADATA_PATH):
        df = pd.read_csv(METADATA_PATH)
        metadata = df.set_index('filename').to_dict('index')
    
    # Process all images in folder
    for root, _, files in os.walk(IMAGE_FOLDER):
        for filename in tqdm(files, desc="Processing images"):
            input_path = os.path.join(root, filename)
            rel_path = os.path.relpath(input_path, IMAGE_FOLDER)
            output_path = os.path.join(OUTPUT_FOLDER, rel_path)
            
            # Create subdirectories if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            try:
                with Image.open(input_path) as img:
                    # Check if this image has special crop parameters
                    crop_params = None
                    if filename in metadata:
                        meta = metadata[filename]
                        microscope = meta.get('microscope')
                        phone = meta.get('phone')
                        
                        # Find matching crop parameters
                        for (m, p), params in CROP_PARAMS.items():
                            if (m == microscope and 
                                (p is None or p == phone)):
                                crop_params = params
                                break
                    
                    if crop_params:
                        # Special cropping case
                        width, height = img.size
                        left = crop_params['left']
                        right = width - crop_params['right']
                        top = crop_params['top']
                        bottom = height - crop_params['bottom']
                        
                        if left < right and top < bottom:
                            img = img.crop((left, top, right, bottom))
                    
                    # Resize all images to target size
                    img = img.resize(TARGET_SIZE, Image.LANCZOS)
                    img.save(output_path)
                    
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                # Copy original as fallback
                shutil.copy2(input_path, output_path)

    print(f"Processing complete! Output saved to {OUTPUT_FOLDER}")

if __name__ == '__main__':
    process_images()