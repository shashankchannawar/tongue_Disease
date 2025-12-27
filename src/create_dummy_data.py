import os
import numpy as np
from PIL import Image

def create_dummy_data(base_dir='data/raw', classes=['Healthy', 'Traumatic Ulcer', 'Candidiasis', 'Leukoplakia', 'Squamous Cell Carcinoma'], num_images=10):
    """
    Creates dummy data for testing the pipeline.
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    for class_name in classes:
        class_dir = os.path.join(base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        print(f"Generating {num_images} images for {class_name}...")
        for i in range(num_images):
            # Generate random image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(os.path.join(class_dir, f'img_{i}.jpg'))
            
    print("Dummy data generation complete.")

if __name__ == "__main__":
    create_dummy_data()
