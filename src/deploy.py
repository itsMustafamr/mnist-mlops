from fastapi import FastAPI, UploadFile, File
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

# Define the Logistic Regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# Load the trained model
model = LogisticRegressionModel(28*28, 10)
model.load_state_dict(torch.load('mnist_logistic_regression.pth'))
model.eval()

# Initialize FastAPI app
app = FastAPI()

# Image preprocessing transformation
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define the prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded file
    image = Image.open(io.BytesIO(await file.read()))
    # Apply preprocessing to the image
    image = transform(image).view(-1, 28 * 28)

    # Make prediction
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()

    return {"prediction": prediction}
