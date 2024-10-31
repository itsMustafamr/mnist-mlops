import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
from sklearn.metrics import precision_score, recall_score
from PIL import Image
import io

# Initialize Weights and Biases (Wandb)
wandb.init(project="mnist-mlops")

# Define the Logistic Regression Model
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        return self.linear(x)

# Hyperparameters
learning_rate = 0.001
num_epochs = 5
batch_size = 64

# Data Loading and Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
val_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, Optimizer, and Loss Function Initialization
model = LogisticRegression()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Watch the model with Wandb
wandb.watch(model, log="all")

# Training Loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log training metrics for each batch
        wandb.log({
            "train_loss": loss.item(),
            "learning_rate": optimizer.param_groups[0]['lr'],
            "gradient_norm": sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        })

    # Validation metrics
    model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            _, preds = torch.max(output, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(target.cpu().numpy())

    precision = precision_score(val_labels, val_preds, average='macro')
    recall = recall_score(val_labels, val_preds, average='macro')

    wandb.log({
        "validation_precision": precision,
        "validation_recall": recall
    })

    # Log sample predictions
    def log_predictions(model, data, target, num_samples=10):
        model.eval()
        with torch.no_grad():
            output = model(data)
            _, preds = torch.max(output, 1)

        # Log sample images with predictions
        for i in range(num_samples):
            wandb.log({ f"sample_{i}": wandb.Image(data[i], caption=f"Pred: {preds[i].item()}, True: {target[i].item()}") })

    log_predictions(model, next(iter(val_loader))[0][:10], next(iter(val_loader))[1][:10])

# Save the trained model
torch.save(model.state_dict(), "src/mnist_model.pth")

# Finish the Wandb run
wandb.finish()

