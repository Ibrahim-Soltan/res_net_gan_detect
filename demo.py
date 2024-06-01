import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# Define the model architecture
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # Load pre-trained ResNet50 model
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Freeze the parameters (weights)
        for param in self.resnet50.parameters():
            param.requires_grad = True

        # Modify the final layer for custom classification
        num_ftrs_in = self.resnet50.fc.in_features  # Get the number of input features from the last layer
        self.resnet50.fc = nn.Linear(num_ftrs_in, 1)  # Replace the fully connected layer
        self.sigmoid = nn.Sigmoid()  # Define sigmoid activation as an attribute

    def forward(self, x):
        x = self.resnet50(x)
        x = self.sigmoid(x)
        return x

# Initialize the model
model = Classifier()

# Load the trained model weights
model.load_state_dict(torch.load('./epoch_6.pth', map_location=torch.device('cpu')))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.CenterCrop(224),  # Optional center crop (adjust size if needed)
    transforms.ToTensor(),  # Convert PIL images to tensors
])

def predict(image):
    image = transform(image).unsqueeze(0)
    output = model(image)
    predicted = (output >= 0.5).float()  # Apply threshold to sigmoid output
    return 'GAN Generated' if predicted.item() == 1 else 'Not GAN Generated'

st.title("GAN Detection Model")
st.write("Upload an image to check if it is GAN-generated or not.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = predict(image)
    st.write(f"Prediction: {label}")
