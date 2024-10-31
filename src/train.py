import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import wandb

# Initialize W&B
wandb.init(project="mnist-mlops", entity="your_wandb_username")

# Define the Logistic Regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# Set input and output dimensions
input_dim = 28 * 28  # MNIST images are 28x28
output_dim = 10  # 10 classes for digits 0-9
model = LogisticRegressionModel(input_dim, output_dim)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 28*28)  # Flatten images

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate training loss
        total_train_loss += loss.item()

        # Log the loss to W&B
        wandb.log({"training_loss": loss.item(), "epoch": epoch + 1})

        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # Log average training loss for the epoch
    avg_train_loss = total_train_loss / len(train_loader)
    wandb.log({"avg_train_loss": avg_train_loss, "epoch": epoch + 1})

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.view(-1, 28*28)  # Flatten images
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate and log accuracy
    accuracy = 100 * correct / total
    wandb.log({"validation_accuracy": accuracy, "epoch": epoch + 1})
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {accuracy:.2f}%')

# Save the trained model
torch.save(model.state_dict(), 'mnist_logistic_regression.pth')

# Finish the W&B session
wandb.finish()
