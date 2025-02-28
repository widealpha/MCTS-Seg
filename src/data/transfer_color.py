import os
from PIL import Image

def convert_to_grayscale(input_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path).convert('L')
            img.save(img_path)
            print(f"Converted {filename} to grayscale.")

def main():
    train_image_dir = 'data/ISIC2016GREY/raw/train/image'
    test_image_dir = 'data/ISIC2016GREY/raw/test/image'
    
    convert_to_grayscale(train_image_dir)
    convert_to_grayscale(test_image_dir)

if __name__ == "__main__":
    main()