import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection._utils import collate_fn

# Custom Dataset Class
class TrafficSignDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        width, height = image.size
        
        # Construct label path
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)
        
        # Check if label file exists
        if not os.path.exists(label_path):
            print(f"Warning: Label file not found for {img_name}. Skipping this image.")
            return None  # Skip this image
        
        with open(label_path, "r") as f:
            lines = f.readlines()
            bboxes = []
            class_labels = []
            for line in lines:
                class_id, x_center, y_center, box_width, box_height = map(float, line.strip().split())
                # Convert YOLO format to pixel coordinates
                x0 = (x_center - box_width / 2) * width
                y0 = (y_center - box_height / 2) * height
                x1 = (x_center + box_width / 2) * width
                y1 = (y_center + box_height / 2) * height
                bboxes.append([x0, y0, x1, y1])
                class_labels.append(int(class_id))
        
        # Convert bboxes to tensor
        boxes = torch.tensor(bboxes, dtype=torch.float32)
        # Convert class labels to tensor
        labels = torch.tensor(class_labels, dtype=torch.int64)
        
        # Create targets dict
        targets = {
            'boxes': boxes,
            'labels': labels
        }
        
        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'targets': targets
        }

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets and dataloaders
train_dataset = TrafficSignDataset(
    image_dir="dataset/images/train",
    label_dir="dataset/labels/train",
    transform=transform
)
# Filter out None values from the dataset
train_dataset = [data for data in train_dataset if data is not None]
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=detection_utils.collate_fn
)

val_dataset = TrafficSignDataset(
    image_dir="dataset/images/val",
    label_dir="dataset/labels/val",
    transform=transform
)
val_dataset = [data for data in val_dataset if data is not None]
val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=detection_utils.collate_fn
)

# Define the model creation function
def get_model(num_classes):
    # Load a pre-trained model for COCO
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Initialize the model
num_classes = 4  # Number of classes (trafficlight, stop, speedlimit, crosswalk)
model = get_model(num_classes)

# Define optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        images = [img.to(device) for img in batch['image']]
        targets = [{k: v.to(device) for k, v in t.items()} for t in batch['targets']]
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        # Backward pass and optimization
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

# Save the model
torch.save(model.state_dict(), "faster_rcnn_traffic_sign_model.pth")
print("Model saved!")