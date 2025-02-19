import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn as nn

# Define CNN Model (Must Match the Trained Model)
class DefectCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(DefectCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load PyTorch Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DefectCNN(num_classes=6).to(device)
model.load_state_dict(torch.load("cnn_defect_model_gpu.pth", map_location=device))
model.eval()

# Define class labels
class_labels = ["Crazing", "Inclusion", "Patches", "Pitted", "Rolled", "Scratches"]

# Streamlit UI
st.title("Coil Surface Defect Detector")
st.write("Upload an image to classify the defect.")

uploaded_file = st.file_uploader("Choose an image...", type=["bmp", "jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert BMP to RGB if necessary
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((128, 128))  # Resize to match model input

    # Convert image to tensor
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension & move to GPU

    # Make prediction
    with torch.no_grad():
        prediction = model(img_tensor)
        predicted_class = class_labels[torch.argmax(prediction).item()]
        confidence = torch.max(torch.nn.functional.softmax(prediction, dim=1)).item() * 100

    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write(f"### Predicted Defect: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
