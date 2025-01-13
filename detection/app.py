import os
import xml.etree.ElementTree as ET

# Directories
annotations_dir = "annotations"  # Folder containing XML files
output_labels_dir = "labels"  # Folder to save YOLO TXT files
os.makedirs(output_labels_dir, exist_ok=True)  # Create the output folder if it doesn't exist

# Class mappings (case-insensitive match)
class_mapping = {
    "trafficlight": 0,
    "stop": 1,
    "speedlimit": 2,
    "crosswalk": 3
}

# Function to convert XML annotations to YOLO format
def convert_annotation(xml_file, output_file):
    try:
        # Parse the XML file
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Get image dimensions
        image_width = int(root.find("size/width").text)
        image_height = int(root.find("size/height").text)

        # Open output file for writing YOLO annotations
        with open(output_file, "w") as f:
            for obj in root.findall("object"):
                class_name = obj.find("name").text.lower()  # Convert to lowercase

                # Skip if the class name is not in the mapping
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

                # Write YOLO format annotation
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    except Exception as e:
        print(f"Error processing {xml_file}: {e}")

# Process each XML file in the annotations folder
for xml_file in os.listdir(annotations_dir):
    if xml_file.endswith(".xml"):
        input_path = os.path.join(annotations_dir, xml_file)
        output_path = os.path.join(output_labels_dir, xml_file.replace(".xml", ".txt"))
        convert_annotation(input_path, output_path)

print("Conversion completed! Check the 'labels' folder for YOLO TXT files.")
