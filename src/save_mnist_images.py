import os
import torch
from torchvision import datasets, transforms
from PIL import Image

# Create a directory to save sample images
os.makedirs('sample_images', exist_ok=True)

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Save the first 10 images as PNG files
for i in range(10):
    image, label = dataset[i]
    image = transforms.ToPILImage()(image)  # Convert tensor to PIL image
    image.save(f'sample_images/mnist_sample_{i}_label_{label}.png')

print("Sample images saved in 'sample_images' directory")
