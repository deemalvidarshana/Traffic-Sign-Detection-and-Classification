import os
import subprocess

# Define paths and parameters
img_size = 640
batch_size = 16
epochs = 50
data_yaml = "data.yaml"
weights = "yolov5s.pt"
project = "C:/Users/deema/Desktop/new/dataset"
name = "traffic-sign-model"

# Step 1: Train the model
train_command = [
    "python", "train.py",
    "--img", str(img_size),
    "--batch", str(batch_size),
    "--epochs", str(epochs),
    "--data", data_yaml,
    "--weights", weights,
    "--project", project,
    "--name", name
]

print("Starting training...")
subprocess.run(train_command)

# Step 2: Evaluate the model
best_weights_path = os.path.join(project, name, "weights", "best.pt")
val_command = [
    "python", "val.py",
    "--data", data_yaml,
    "--weights", best_weights_path,
    "--img", str(img_size)
]

print("Starting evaluation...")
subprocess.run(val_command)

print("Training and evaluation completed!")