import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops.layers.torch import Rearrange
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from models.models.create_models import ImageClassifier
import os
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
# model_name = "mobilenetv2_100" # resnext50_32x4d resnet18 mobilenetv2_100 tf_efficientnet_b0

dataset_root  = r"C:\dataset"
model_name = "mobilenetv2_100"
no_classes = 11
# model = ImageClassifier(no_classes, model_name).cuda()
model = ImageClassifier(no_classes, model_name)

# Instantiate the model
image_size = 224

num_epochs = 100
batch_size = 4
# model = model.cuda()
model = model
# print(model)
# model = ViT(image_size, patch_size, stride, num_classes, dim, depth, num_heads, local_context_size)

# Training code
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),  # Resize the images to a specific size
    transforms.ToTensor(),  # Convert the images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
])

# Define the path to the root folder of the dataset


# Create an instance of the ImageFolder dataset for the training data
train_dataset = ImageFolder(root=dataset_root + "/train", transform=transform)

# Create an instance of the ImageFolder dataset for the validation data
validation_dataset = ImageFolder(root=dataset_root + "/test", transform=transform)

# Set batch size for dataloaders


# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=False)
# criterion = criterion.cuda()
criterion = criterion

training_loss_arr = []
validation_loss_arr = []

# Assuming you have a dataloader for your dataset
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for images, labels in train_dataloader:
        optimizer.zero_grad()
        # images = images.cuda()
        # labels = labels.cuda()
        images = images
        labels = labels

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_dataloader)
    training_loss_arr.append(avg_train_loss)
    # Validation
    model.eval()
    total_val_loss = 0.0
    for images, labels in validation_dataloader:
        with torch.no_grad():
            # images = images.cuda()
            # labels = labels.cuda()

            images = images
            labels = labels
            outputs = model(images)
            val_loss = criterion(outputs, labels)
            total_val_loss += val_loss.item()
    avg_val_loss = total_val_loss / len(validation_dataloader)
    validation_loss_arr.append(avg_val_loss)
    # Print the average loss for the epoch
    print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

         # Print loss for monitoring training progress
        # print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, loss.item()))

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), "{}_{}.pth".format(model_name, epoch))
        print("Model saved successfully.")
        # Validation code
        model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in test_dataloader:
                # images = images.cuda()
                # labels = labels.cuda()
                images = images
                labels = labels
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

        accuracy = 100 * total_correct / total_samples
        print("Validation Accuracy: {:.2f}%".format(accuracy))

print(training_loss_arr)
print(validation_loss_arr)

df = pd.DataFrame({'train_loss':training_loss_arr, 'val_loss':validation_loss_arr})
df.to_csv('C:trainandvallossagain.csv', index = False)

# Plot loss curves
plt.figure(figsize=(8, 4), dpi=300)  # Create a figure with the specified size and resolution
plt.plot(training_loss_arr, label="Training Loss")
plt.plot(validation_loss_arr, label="Validation Loss")
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Loss Value', fontsize=12, fontweight='bold')
plt.title('Loss curve', fontsize=14, fontweight='bold')
plt.legend()

# Save the plot as an EPS file
plt.savefig('Loss Curve.eps', format='eps', bbox_inches='tight')
plt.show()


# Save the trained model
torch.save(model.state_dict(), "vit_model.pth")
print("Model saved successfully.")
