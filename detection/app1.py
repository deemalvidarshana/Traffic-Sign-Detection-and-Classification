import os
import random
import shutil
import xml.etree.ElementTree as ET

# Directories
images_dir = "images"  # Folder with original images
annotations_dir = "annotations"  # Folder with XML files
dataset_dir = "dataset"  # Output dataset directory
os.makedirs(dataset_dir, exist_ok=True)

# Create output folders for train and val splits
split_dirs = ["images/train", "images/val", "labels/train", "labels/val"]
for split_dir in split_dirs:
    os.makedirs(os.path.join(dataset_dir, split_dir), exist_ok=True)

# Class mappings (case-insensitive)
class_mapping = {
    "trafficlight": 0,
    "stop": 1,
    "speedlimit": 2,
    "crosswalk": 3
}

# Function to convert XML annotations to YOLO format
def convert_annotation(xml_file, output_file, image_width, image_height):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        with open(output_file, "w") as f:
            for obj in root.findall("object"):
                class_name = obj.find("name").text.lower()

                # Skip if class is not in mapping
                if class_name not in class_mapping:
                    print(f"Warning: Class '{class_name}' not found in class_mapping. Skipping...")
                    continue

                class_id = class_mapping[class_name]
                bndbox = obj.find("bndbox")
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)

                # Convert to YOLO format
                x_center = ((xmin + xmax) / 2) / image_width
                y_center = ((ymin + ymax) / 2) / image_height
                width = (xmax - xmin) / image_width
                height = (ymax - ymin) / image_height

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    except Exception as e:
        print(f"Error processing {xml_file}: {e}")

# Collect all images and corresponding XML files
image_files = [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png"))]
xml_files = [f.replace(".jpg", ".xml").replace(".png", ".xml") for f in image_files]

# Split dataset into train (80%) and val (20%)
random.seed(42)  # For reproducibility
data = list(zip(image_files, xml_files))
random.shuffle(data)
split_idx = int(0.8 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]

# Process and organize dataset
for split, split_data in zip(["train", "val"], [train_data, val_data]):
    for image_file, xml_file in split_data:
        image_path = os.path.join(images_dir, image_file)
        xml_path = os.path.join(annotations_dir, xml_file)

        # Skip if corresponding XML file doesn't exist
        if not os.path.exists(xml_path):
            print(f"Warning: Annotation {xml_file} not found for {image_file}. Skipping...")
            continue

        # Read image dimensions
        from PIL import Image
        with Image.open(image_path) as img:
            image_width, image_height = img.size

        # Convert XML to YOLO TXT
        label_path = os.path.join(dataset_dir, f"labels/{split}/{image_file.replace('.jpg', '.txt').replace('.png', '.txt')}")
        convert_annotation(xml_path, label_path, image_width, image_height)

        # Copy image to appropriate split folder
        shutil.copy(image_path, os.path.join(dataset_dir, f"images/{split}/{image_file}"))

print("Dataset organized successfully!")
