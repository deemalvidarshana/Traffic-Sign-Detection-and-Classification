import cv2
import torch
import time
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# Load the YOLOv5 model
yolov5_model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/deema/Desktop/new/dataset/traffic-sign-model/weights/best.pt')
yolov5_model.conf = 0.70  # Adjust confidence threshold as needed

# Load the ResNet50 model
resnet_model_path = r"C:\Users\deema\Desktop\new\model\resnet50_classification.pth"
resnet_model = models.resnet50(pretrained=False)  # Load the ResNet50 architecture
num_classes = 4  # Update this based on your number of classes
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)  # Modify the final layer
resnet_model.load_state_dict(torch.load(resnet_model_path))  # Load the trained weights
resnet_model.eval()  # Set the model to evaluation mode

# Move the models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolov5_model = yolov5_model.to(device)
resnet_model = resnet_model.to(device)

# Define the classes (update based on your model classes)
yolov5_classes = ['Traffic Light', 'Stop', 'Speed Limit', 'Crosswalk']
resnet_classes = ['Traffic Light', 'Stop', 'Speed Limit', 'Crosswalk']

# Define the image preprocessing pipeline for ResNet
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input size
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Initialize the webcam (use 0 for default camera)
cap = cv2.VideoCapture(0)  # Replace 0 with the video file path if needed

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Real-time video feed processing
try:
    while True:
        start_time = time.time()

        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert frame to RGB (YOLOv5 expects RGB images)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform object detection with YOLOv5
        yolov5_results = yolov5_model(rgb_frame)
        detections = yolov5_results.pandas().xyxy[0]  # Bounding boxes, confidence, and class labels

        # Process each detection
        for _, row in detections.iterrows():
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            conf = row['confidence']
            cls = int(row['class'])
            label = f"{yolov5_classes[cls]} {conf:.2f}"

            # Draw the bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Crop the detected object for classification
            detected_object = frame[ymin:ymax, xmin:xmax]
            if detected_object.size == 0:
                continue  # Skip empty detections

            # Convert the cropped image to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(detected_object, cv2.COLOR_BGR2RGB))

            # Preprocess the image for ResNet
            input_tensor = resnet_transform(pil_image).unsqueeze(0)  # Add batch dimension
            input_tensor = input_tensor.to(device)  # Move to GPU if available

            # Perform classification with ResNet
            with torch.no_grad():
                outputs = resnet_model(input_tensor)
                _, predicted = torch.max(outputs, 1)
                class_id = predicted.item()
                class_name = resnet_classes[class_id]

            # Add the classification result to the label
            label += f" ({class_name})"

            # Add the label to the frame
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Traffic Sign Detection and Classification', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Calculate FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        print(f"FPS: {fps:.2f}")

finally:
    # Release the capture and close the windows
    cap.release()
    cv2.destroyAllWindows()