import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import os
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Dataset path and model configuration
dataset_root  = r"dataset_sample"
model_name = "mobilenetv2_100"
no_classes = 5
image_size = 224
num_epochs = 200
batch_size = 4
learning_rate = 0.001

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Definition
class ImageClassifier(nn.Module):
    def __init__(self, num_classes, model_name):
        super(ImageClassifier, self).__init__()
        if model_name == "mobilenetv2_100":
            self.model = models.mobilenet_v2(weights=True)
            self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def forward(self, x):
        return self.model(x)

# Instantiate the model
model = ImageClassifier(no_classes, model_name).to(device)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = ImageFolder(root=os.path.join(dataset_root, "train"), transform=transform)
validation_dataset = ImageFolder(root=os.path.join(dataset_root, "test"), transform=transform)

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

# Training and Validation
training_loss_arr = []
validation_loss_arr = []
training_acc_arr = []
validation_acc_arr = []

for epoch in range(num_epochs):
    # Training loop
    model.train()
    total_train_loss = 0.0
    total_train_correct = 0
    total_train_samples = 0
    
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train_correct += (predicted == labels).sum().item()
        total_train_samples += labels.size(0)

    avg_train_loss = total_train_loss / len(train_dataloader)
    training_loss_arr.append(avg_train_loss)
    
    train_accuracy = total_train_correct / total_train_samples * 100
    training_acc_arr.append(train_accuracy)

    # Validation loop
    model.eval()
    total_val_loss = 0.0
    total_val_correct = 0
    total_val_samples = 0

    with torch.no_grad():
        for images, labels in validation_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss = criterion(outputs, labels)
            total_val_loss += val_loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_val_correct += (predicted == labels).sum().item()
            total_val_samples += labels.size(0)

    avg_val_loss = total_val_loss / len(validation_dataloader)
    validation_loss_arr.append(avg_val_loss)

    val_accuracy = total_val_correct / total_val_samples * 100
    validation_acc_arr.append(val_accuracy)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%")

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"{model_name}_epoch{epoch+1}.pth")
        print(f"Model saved at epoch {epoch+1}.")

# Save the final model
torch.save(model.state_dict(), f"{model_name}_final.pth")
print(f"Final model saved as {model_name}_final.pth.")

# Save loss and accuracy data to CSV
df = pd.DataFrame({'train_loss': training_loss_arr, 'val_loss': validation_loss_arr,
                   'train_acc': training_acc_arr, 'val_acc': validation_acc_arr})
df.to_csv('train_val_loss_acc.csv', index=False)

# Plot loss and accuracy curves
plt.figure(figsize=(12, 6), dpi=300)

# Plot loss curves
plt.subplot(1, 2, 1)
plt.plot(training_loss_arr, label="Training Loss", color='b')
plt.plot(validation_loss_arr, label="Validation Loss", color='r')
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Loss Value', fontsize=12, fontweight='bold')
plt.title('Loss Curve', fontsize=14, fontweight='bold')
plt.legend()

# Plot accuracy curves
plt.subplot(1, 2, 2)
plt.plot(training_acc_arr, label="Training Accuracy", color='b')
plt.plot(validation_acc_arr, label="Validation Accuracy", color='r')
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('Accuracy Curve', fontsize=14, fontweight='bold')
plt.legend()

# Save the plots
plt.savefig('loss_acc_curve.eps', format='eps', bbox_inches='tight')
plt.show()
