**MobileNet** is a family of deep learning models designed specifically for efficient execution on mobile and embedded devices. The primary goal of MobileNet architectures is to achieve a balance between model size, computational complexity, and accuracy, making them well-suited for applications where computational resources and power consumption are limited.

### Key Features of MobileNet:

#### Depthwise Separable Convolutions:

The primary innovation in MobileNet is the use of depthwise separable convolutions, which reduce the computational cost and model size significantly.

- A standard convolution operation is divided into two parts:
  - **Depthwise Convolution:** Applies a single convolutional filter per input channel.
  - **Pointwise Convolution:** Uses a 1x1 convolution to combine the outputs of the depthwise convolution.

#### Model Variants:

- **MobileNetV1:** Introduced the concept of depthwise separable convolutions, resulting in significant reductions in the number of parameters and computational cost compared to traditional convolutional networks.
- **MobileNetV2:** Improved upon V1 by introducing inverted residuals and linear bottlenecks, enhancing both efficiency and performance.
- **MobileNetV3:** Further optimized using a combination of platform-aware neural architecture search (NAS) and a novel network structure, balancing accuracy and efficiency even more effectively.

#### Width Multiplier:

MobileNet models include a width multiplier parameter (α), which scales the number of channels in each layer, allowing for the adjustment of the model size and computational cost.

- α values range from 0 to 1, where lower values reduce the model size and computational load.

#### Resolution Multiplier:

MobileNet allows the input image resolution to be reduced, further decreasing the computational complexity.

#### Applications:

MobileNets are widely used in mobile and embedded vision applications, such as image classification, object detection, and face recognition, due to their efficient architecture.

### Summary:

MobileNet models are highly efficient convolutional neural networks optimized for mobile and embedded devices. They achieve significant reductions in model size and computational complexity through the use of depthwise separable convolutions, making them ideal for resource-constrained environments. Various versions of MobileNet (V1, V2, V3) offer different levels of optimization, allowing users to select the appropriate balance of efficiency and performance for their specific application.

This code is a script for training an image classification model MobileNet using PyTorch. The code is structured to facilitate easy training, evaluation, and monitoring of a deep learning model for image classification. It allows for periodic saving of the model's state and provides insights into the model's performance through loss curves and accuracy metrics.

Here is a step-by-step summary of the code:
#### 1.  Import Libraries
- **Libraries:** Import necessary libraries including PyTorch, torchvision, einops, and other utility libraries such as matplotlib and pandas.
```ruby
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops.layers.torch import Rearrange
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import os
import pandas as pd
from models.models.create_models import ImageClassifier
```
#### 2. Set Environment Variables
- Set environment variables to avoid certain errors.
```ruby
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
```
#### 3. Configure Model and Dataset Paths
- Define the root directory for the dataset and the model name. Specify the number of classes and instantiate the model.
```ruby
dataset_root = r"C:\dataset"
model_name = "mobilenetv2_100"
no_classes = 11
model = ImageClassifier(no_classes, model_name)
```
#### 4. Set Training Parameters
- Define training parameters such as image size, number of epochs, and batch size.
```ruby
image_size = 224
num_epochs = 100
batch_size = 4
```
#### 5. Set Up Optimizer and Loss Function
- Configure the optimizer and the loss function for training.
```ruby
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
```
#### 6. Define Data Transformations
- Define transformations to be applied to the images such as resizing, tensor conversion, and normalization.
```ruby
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```
#### 7. Create Datasets and DataLoaders
- Create datasets and dataloaders for training, validation, and testing.
```ruby
train_dataset = ImageFolder(root=dataset_root + "/train", transform=transform)
validation_dataset = ImageFolder(root=dataset_root + "/test", transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=False)
```
#### 8. Train the Model
- Train the model for a specified number of epochs, compute training and validation loss, and save the model every 10 epochs.
```ruby
training_loss_arr = []
validation_loss_arr = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for images, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_dataloader)
    training_loss_arr.append(avg_train_loss)

    model.eval()
    total_val_loss = 0.0
    for images, labels in validation_dataloader:
        with torch.no_grad():
            outputs = model(images)
            val_loss = criterion(outputs, labels)
            total_val_loss += val_loss.item()
    avg_val_loss = total_val_loss / len(validation_dataloader)
    validation_loss_arr.append(avg_val_loss)
    print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), "{}_{}.pth".format(model_name, epoch))
        print("Model saved successfully.")
 ```       
#### 9. Validate the Model
- Validate the model after every 10 epochs and calculate the accuracy.
```ruby
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in test_dataloader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

        accuracy = 100 * total_correct / total_samples
        print("Validation Accuracy: {:.2f}%".format(accuracy))
```  
#### 10. Save Training and Validation Loss
- Save the training and validation loss to a CSV file.
```ruby
df = pd.DataFrame({'train_loss': training_loss_arr, 'val_loss': validation_loss_arr})
df.to_csv('C:/trainandvallossagain.csv', index=False)
```
#### 11. Plot Loss Curves
- Plot the training and validation loss curves and save the plot as an EPS file.
```ruby
plt.figure(figsize=(8, 4), dpi=300)
plt.plot(training_loss_arr, label="Training Loss")
plt.plot(validation_loss_arr, label="Validation Loss")
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Loss Value', fontsize=12, fontweight='bold')
plt.title('Loss curve', fontsize=14, fontweight='bold')
plt.legend()
plt.savefig('Loss Curve.eps', format='eps', bbox_inches='tight')
plt.show()
```
#### 12. Save the Trained Model
- Save the final trained model's state dictionary.
```ruby
torch.save(model.state_dict(), "vit_model.pth")
print("Model saved successfully.")
```
# Generate Continuous Wavelet Transform (CWT) spectrograms

### Purpose of the Script
This script is designed to process time series data from CSV files and generate Continuous Wavelet Transform (CWT) spectrograms for each column in the data. It's typically used for signal analysis, particularly in scenarios involving time-frequency representation of signals.

Let me provide a deeper explanation of the Continuous Wavelet Transform (CWT) and its significance in signal processing.

### Continuous Wavelet Transform (CWT) - Detailed Explanation

#### What is a Wavelet Transform?
A wavelet transform is a technique for analyzing signals by breaking them down into wavelets (small wave-like oscillations). Unlike Fourier transforms that use sine waves of fixed length, wavelets can adapt their length to examine different parts of a signal.

#### Key Characteristics of CWT
1. **Time-Frequency Representation**
   - Provides a way to analyze how a signal's frequency content changes over time
   - Ideal for non-stationary signals that have changing characteristics

2. **Wavelet Types**
   - In this script, a Morlet wavelet is used (specified by `wavelet=mother`)
   - Morlet wavelet is particularly good for:
     - Analyzing oscillatory signals
     - Providing good time and frequency localization
     - Resembling sine waves with a Gaussian envelope

#### Mathematical Concept
The CWT is mathematically defined as:
```
W(a,b) = ∫ f(t) * ψ*((t-b)/a) dt
```
Where:
- `f(t)` is the original signal
- `ψ` is the wavelet function
- `a` is the scale (related to frequency)
- `b` is the translation (time location)

#### Practical Implementation in the Script
```python
mother = wavelet.Morlet(6)
wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(y, 1 / sampling_rate, wavelet=mother)
```
Breaking down the parameters:
- `y`: Input signal
- `1 / sampling_rate`: Time resolution
- `wavelet=mother`: Specifies the wavelet type (Morlet)

#### Spectrogram Generation
```python
plt.imshow(np.abs(wave), extent=[x[0], x[-1], freqs[-1], freqs[0]], cmap='jet', aspect='auto')
```
This line creates a visual representation where:
- X-axis: Time
- Y-axis: Frequency
- Color intensity: Wavelet coefficient magnitude
- `cmap='jet'`: Color scheme (blue to red)

### Practical Applications
1. **Biomedical Signals**
   - EEG/ECG analysis
   - Detecting abnormal patterns

2. **Geophysics**
   - Seismic signal processing
   - Earthquake data analysis

3. **Audio Processing**
   - Music analysis
   - Sound event detection

4. **Mechanical Engineering**
   - Fault detection in machinery
   - Vibration analysis

### Advantages Over Traditional Fourier Transform
- Better time localization
- Can detect transient events
- Handles non-stationary signals more effectively
- Provides multi-resolution analysis

### Potential Improvements for This Script
1. Add more wavelet options
2. Implement dynamic scaling
3. Add signal preprocessing
4. Create more comprehensive visualization options

### Detailed Script Breakdown

#### 1. Import Statements
```python
import os           # For file and directory operations
import numpy as np  # For numerical computations
import pandas as pd # For data manipulation
import matplotlib.pyplot as plt  # For plotting
import pycwt as wavelet  # Wavelet transformation library
```
These libraries are imported to handle various tasks:
- `os`: Managing file paths and directories
- `numpy`: Numerical computations
- `pandas`: Reading and processing CSV data
- `matplotlib`: Creating and saving plots
- `pycwt`: Performing Continuous Wavelet Transform

#### 2. Path Configuration
```python
base_csv_path = r"C:\CSVs\Seperate"     # Directory containing input CSV files
base_output_path = r"C:\Spectrogram\5pix"  # Directory to save output spectrograms
```
Sets up base directories for input CSV files and output spectrogram images.

#### 3. Class Generation
```python
classes = [f"C{i}" for i in range(1, 10)]
```
Creates a list of classes from C1 to C9, which will be used to process corresponding CSV files.

#### 4. CWT Spectrogram Processing Function
```python
def process_cwt_spectrogram(data, output_folder, sampling_rate=10048):
```
This function does the core processing:
- Removes the first column of the input data
- Iterates through each column in the DataFrame
- Performs Continuous Wavelet Transform
- Generates and saves a spectrogram for each column

Key steps within the function:
- Extract signal values from each column
- Generate time axis based on sampling rate
- Use Morlet wavelet for transformation
- Create a spectrogram image using `plt.imshow()`
- Save the image with specific formatting (5 DPI, tight layout, no borders)

#### 5. Main Processing Loop
```python
for class_name in classes:
```
The main processing loop:
- Constructs full paths for input CSV and output folder
- Creates output directory if it doesn't exist
- Attempts to:
  1. Read the CSV file
  2. Process the data using `process_cwt_spectrogram()`
  3. Print processing status
- Handles potential errors like missing files

### Workflow
1. Script starts by defining input and output paths
2. Generates a list of classes (C1 to C9)
3. For each class:
   - Finds corresponding CSV file
   - Creates an output folder
   - Reads the CSV
   - Generates CWT spectrograms for each column
   - Saves spectrograms as PNG images

### Technical Details
- Uses Continuous Wavelet Transform for time-frequency analysis
- Morlet wavelet used for transformation
- Spectrograms saved at very low resolution (5 DPI)
- Handles potential file not found or processing errors

### Potential Use Cases
- Signal processing
- Time series analysis
- Feature extraction for machine learning
- Visualizing frequency components of signals

### Recommendations for Improvement
1. Add more error handling
2. Make sampling rate configurable
3. Add logging instead of print statements
4. Parameterize wavelet type and parameters



<div align="center">
  <a href="https://maazsalman.org/">
    <img width="70" src="click-svgrepo-com.svg" alt="gh" />
  </a>
  <p> Explore More! 🚀</p>
</div>
