import cv2
import numpy as np
import torch
import pickle
from torchvision import transforms
from PIL import Image

# Constants
frameWidth = 640         # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
font = cv2.FONT_HERSHEY_SIMPLEX

# Define the classes for YOLOv5 (these should correspond to your model's output classes)
yolov5_classes = ['Traffic Light', 'Stop', 'Speed Limit', 'Crosswalk']  # Update based on your YOLOv5 model's classes

# Setup the video camera
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# Load YOLOv5 model
yolov5_model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
yolov5_model.conf = 0.70  # Adjust confidence threshold as needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolov5_model = yolov5_model.to(device)

# Load the trained classification model
pickle_in = open("model_trained.p", "rb")  # Read byte mode
classification_model = pickle.load(pickle_in)

# Define the preprocessing function for classification
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255  # Normalize
    return img

def getClassName(classNo):
    if classNo == 0:
        return 'trafficlight'
    elif classNo == 1:
        return 'stop'
    elif classNo == 2:
        return 'speedlimit'
    elif classNo == 3:
        return 'crosswalk'

# Define the image preprocessing pipeline for classification model
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

while True:
    # Capture frame-by-frame
    success, imgOrignal = cap.read()
    if not success:
        print("Error: Could not read frame.")
        break

    # Convert frame to RGB (YOLOv5 expects RGB images)
    rgb_frame = cv2.cvtColor(imgOrignal, cv2.COLOR_BGR2RGB)

    # Perform object detection with YOLOv5 (independent of classification)
    yolov5_results = yolov5_model(rgb_frame)
    detections = yolov5_results.pandas().xyxy[0]  # Bounding boxes, confidence, and class labels

    # Process each detection for bounding boxes (YOLOv5)
    for _, row in detections.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        conf = row['confidence']
        cls = int(row['class'])
        label = f"{yolov5_classes[cls]} {conf:.2f}"  # YOLOv5 class name and confidence

        # Draw the bounding box for the detected object in red (YOLOv5)
        cv2.rectangle(imgOrignal, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        # Crop the detected object for classification (independent of detection)
        detected_object = imgOrignal[ymin:ymax, xmin:xmax]
        if detected_object.size == 0:
            continue  # Skip empty detections

        # Preprocess the image for classification (this happens independently of YOLO)
        img = np.asarray(detected_object)
        img = cv2.resize(img, (32, 32))
        img = preprocessing(img)
        img = img.reshape(1, 32, 32, 1)

        # Predict the class with the classification model
        predictions = classification_model.predict(img)
        classIndex = np.argmax(predictions, axis=1)
        probabilityValue = np.amax(predictions)

        # Show the predicted class and probability in green
        class_name = getClassName(classIndex[0])
        cv2.putText(imgOrignal, f"{class_name} {round(probabilityValue * 100, 2)}%", (xmin, ymin - 10), font, 0.5, (0, 255, 0), 2)

    # Display the YOLOv5 class and confidence score in red (for detection)
    for _, row in detections.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        conf = row['confidence']
        cls = int(row['class'])

    # Display the resulting image with detection (red) and classification (green)
    cv2.imshow("Result", imgOrignal)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()
