import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import io
from fastapi import FastAPI, UploadFile, File
import wandb

# Initialize Weights and Biases (Wandb)
wandb.init(project="mnist-mlops")

# Define the Logistic Regression Model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# Load the trained model
model = LogisticRegressionModel(28 * 28, 10)
model.load_state_dict(torch.load('src/mnist_model.pth'))
model.eval()

# Initialize FastAPI app
app = FastAPI()

# Image preprocessing transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure image is grayscale
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define the prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        image = Image.open(io.BytesIO(await file.read()))

        # Apply preprocessing to the image
        image = transform(image).view(-1, 28 * 28)

        # Make prediction
        with torch.no_grad():
            output = model(image)
            prediction = torch.argmax(output, dim=1).item()

        return {"prediction": prediction}

    except Exception as e:
        return {"error": str(e)}
