import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms

# Load your trained PyTorch model
model = torch.load('./epoch_6.pth', map_location=torch.device('cpu'))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.CenterCrop(224),  # Optional center crop (adjust size if needed)
    transforms.ToTensor(),  # Convert PIL images to tensors
])

def predict(image):
    image = transform(image).unsqueeze(0)
    output = model(image)
    _, predicted = torch.max(output, 1)
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
