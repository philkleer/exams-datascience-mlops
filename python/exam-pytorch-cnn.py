# CNN and PyTorch Basics

# The exam will consist of a first part validating basic knowledge on tensors and gradients. The second part will involve solving a classification problem, followed by interpreting the trained model.

# Exercise 1: Tensor and Gradient

# Create a tensor x that will track the gradient and initialize it with a 3x3 matrix filled with ones.

import torch

x = torch.ones((3, 3), requires_grad=True)

print(x)

# Initialize two new variables f and g and assign them the values of the following functions of x.
# \begin{align} f(x) = \cos(x^2) \\ g(x) = \tan(x)^{\frac{1}{4}} \end{align}
# Define the variable h as follows:
# \begin{equation} h(x) = \frac{1}{1+\exp(f(x) - g(x))} \end{equation}
# Ensure gradient computation for f and g.

f = torch.cos(x**2)
g = torch.tan(x**0.25)
f.retain_grad()
g.retain_grad()

h = 1 / (1 + torch.exp(f - g))

# Add the elements of h and store them in a variable s.

# Compute the gradient of s with respect to x.

# Verify the following:

# \begin{align} \frac{\partial s}{\partial x} \Bigr| _{x = 1} &= 0.5291 \\ \frac{\partial s}{\partial f} \Bigr| _{x = 1} &= -0.2303 \\ \frac{\partial s}{\partial g} \Bigr| _{x = 1} &= 0.2303 \end{align}

s = h.sum()

s.backward()

print("partial s / partial x at x-1: ", x.grad)
print("partial s / partial f at x-1: ", f.grad)
print("partial s / partial g at x-1: ", g.grad)

# Exercise 2: Separate Ants and Bees

# Your task in this problem is to train a convolutional neural network (CNN) to distinguish ants from bees. You will use the ResNet50 pretrained model from TorchVision, which you will retrain and interpret.

# The images are in the datasets/train and datasets/valid folders, organized in the following structure:

# root_dir/
# ├── class1/ 
# │ ├── image1.png 
# │ ├── image2.png 
# │ └── ... 
# └── class2/
#   ├── image1.png 
#   ├── image2.png
#   └── ... ...

# Run the following cell to display the data structure.
import os

def display_directory_structure_limited(root_dir, max_files=3):
    for root, dirs, files in os.walk(root_dir):
        level = root.replace(root_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for i, f in enumerate(files):
            if i < max_files:
                print(f"{sub_indent}{f}")
            else:
                print(f"{sub_indent}...")
                break


display_directory_structure_limited('datasets')

# Generator Definition

# Define a transform object that resizes images to (224, 244), applies the same preprocessing operations as the resnet50 model, and converts images to tensors.

# Define a dataset_train and a dataset_test using ImageFolder corresponding to the 'datasets/train' and 'datasets/valid' folders, respectively.

# Define a dataloader_train and a dataloader_test that will group the data in batches of 32.

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets with transform from above
dataset_train = datasets.ImageFolder(root='datasets/train', transform=transform)
dataset_test = datasets.ImageFolder(root='datasets/valid', transform=transform)

# create data loader (batches)
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=True)

# Run the following cell to display a few images from the first data batch.
import torch
import torchvision
import matplotlib.pyplot as plt

def show_images(dataloader):
    for images, labels in dataloader:
        fig, axes = plt.subplots(1, 6, figsize=(12,12))
        for i, img in enumerate(images[:6]):
            axes[i].imshow(img.permute(1, 2, 0))
        break

show_images(dataloader_train)

# Model

# Load the resnet50 model under the name model.

# Freeze the model.

# How many neurons do the last layer have, and what is its name?

from torchvision import models
import torch.nn as nn

model = models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad=False

last_layer = model.fc
num_neurons = last_layer.out_features

print('Last layer name is "fc"')
print(f'Number of neurons in last layer: {num_neurons}')

# Adapt the model to our problem. As a reminder, this is a two-class problem.

# Make sure the model runs on the "CPU".

# Also check that the layer has been modified by displaying the model structure.

device = "cpu"

num_classes = 2
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

from torchsummary import summary
summary(model, input_size=((3, 224, 224)), device=device)

# Define a relevant loss function under the name criterion.

# To test the loss function on a batch of data, generate the model output for X_batch, then use the loss function between the prediction and y_batch. Be careful to switch the tensors to "cpu".

X_batch, y_batch = next(iter(dataloader_train))

criterion = nn.CrossEntropyLoss()

model.eval()

with torch.no_grad():
    y_pred = model(X_batch)
    
loss = criterion(y_pred, y_batch)

print(f'Loss: {loss.item()}')

# Model Traning

# Train the model for a few epochs. Show for each epoch the train loss function, as well as the accuracy on the test dataset.
epochs = 5
num_classes = 2
batch_size = 32

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-1)

for epoch in range(epochs):
    # In this mode some layers of the model act differently
    model.train()
    loss_total = 0
    
    for X_batch, y_batch in dataloader_train:
        optimizer.zero_grad()
        
        y_pred = model(X_batch)
        
        loss = criterion(y_pred, y_batch)
        
        loss.backward()
        
        optimizer.step()
        
        loss_total += loss.item()
    
    avg_train_loss = loss_total / len(dataloader_train)
    
    model.eval()
    correct = 0
    total = 0 
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader_test:
            y_pred = model(X_batch)
            _, predicted = torch.max(y_pred, 1)
            
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
    test_accuracy = (correct/total)
    
    print(f'Epoch: {epoch+1}/{epochs}')
    print(f'Training loss: {avg_train_loss:.4f}')
    print(f'Test accuracy: {test_accuracy:.4f}')

# Unfreeze the parameters of the model.layer4 block.
for param in model.parameters():
    param.requires_grad=False
    
for param in model.layer4.parameters():
    param.requires_grad=True
    
# optimizer and epochs as above
for epoch in range(epochs):
    # In this mode some layers of the model act differently
    model.train()
    loss_total = 0
    
    for X_batch, y_batch in dataloader_train:
        optimizer.zero_grad()
        
        y_pred = model(X_batch)
        
        loss = criterion(y_pred, y_batch)
        
        loss.backward()
        
        optimizer.step()
        
        loss_total += loss.item()
    
    avg_train_loss = loss_total / len(dataloader_train)
    
    model.eval()
    correct = 0
    total = 0 
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader_test:
            y_pred = model(X_batch)
            _, predicted = torch.max(y_pred, 1)
            
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
    test_accuracy = (correct/total)
    
    print(f'Epoch: {epoch+1}/{epochs}')
    print(f'Training loss: {avg_train_loss:.4f}')
    print(f'Test accuracy: {test_accuracy:.4f}')

# Retrain the model with a smaller learning rate over a few epochs, displaying the different metrics for each metric as before.
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)

for param in model.parameters():
    param.requires_grad=False
    
for param in model.layer4.parameters():
    param.requires_grad=True
    
# optimizer and epochs as above
for epoch in range(epochs):
    # In this mode some layers of the model act differently
    model.train()
    loss_total = 0
    
    for X_batch, y_batch in dataloader_train:
        optimizer.zero_grad()
        
        y_pred = model(X_batch)
        
        loss = criterion(y_pred, y_batch)
        
        loss.backward()
        
        optimizer.step()
        
        loss_total += loss.item()
    
    avg_train_loss = loss_total / len(dataloader_train)
    
    model.eval()
    correct = 0
    total = 0 
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader_test:
            y_pred = model(X_batch)
            _, predicted = torch.max(y_pred, 1)
            
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
    test_accuracy = (correct/total)
    
    print(f'Epoch: {epoch+1}/{epochs}')
    print(f'Training loss: {avg_train_loss:.4f}')
    print(f'Test accuracy: {test_accuracy:.4f}')

# We can see that the mechanism is not working perfectly good. If the animal is big (pi

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image

def make_gradcam_heatmap(img_tensor, model, target_layer_name, pred_index=None):
    # Skip forward to get target layer predictions and activations
    def forward_hook(module, input, output):
        model.features = output

    # Register a hook on the target layer to retrieve its outputs
    hook = model._modules.get(target_layer_name).register_forward_hook(forward_hook)
    
    # Perform a forward pass to obtain model outputs
    output = model(img_tensor)
    
    # Remove hook after getting activations
    hook.remove()

    # If no prediction index is provided, use the one with the highest probability
    if pred_index is None:
        pred_index = output.argmax(dim=1).item()
    
    # Class probability value
    y = output[0, pred_index]
    
    # Skip backwards to get the gradients of the target layer
    model.zero_grad() # Reset gradients
    model.features.retain_grad() # Keep target layer gradients
    y.backward(retain_graph=True) # Calculate gradients by backpropagation

    # Get target layer gradients and activations
    gradients = model.features.grad[0]
    activations = model.features[0]

    # Apply average global pooling on gradients
    pooled_grads = torch.mean(gradients, dim=[1, 2])

    # Weight activations by gradients
    for i in range(len(pooled_grads)):
        activations[i, :, :] *= pooled_grads[i]

    # Calculate the heatmap
    heatmap = torch.mean(activations, dim=0).detach().numpy()
    heatmap = np.maximum(heatmap, 0) # Keep only positive values
    heatmap /= np.max(heatmap) # Normalize the heatmap

    return heatmap

target_layer = 'layer4'

img_paths = [
    'datasets/valid/bees/2173503984_9c6aaaa7e2.jpg',
    'datasets/valid/bees/54736755_c057723f64.jpg',
    'datasets/valid/ants/459442412_412fecf3fe.jpg',
    'datasets/valid/ants/892676922_4ab37dce07.jpg'
]

bees_index = 1
ants_index = 0

preprocess = transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


for img_path in img_paths:
    img = Image.open(img_path)
    img_tensor = preprocess(img).unsqueeze(0)
    
    heatmap_bees = make_gradcam_heatmap(img_tensor, model, target_layer, bees_index)
    heatmap_ants = make_gradcam_heatmap(img_tensor, model, target_layer, ants_index)
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    #plt.title(f'Original image: {img_path.split('/')[-1]}')
    
    plt.subplot(2, 2, 2)
    plt.imshow(heatmap_bees, cmap='jet')
    plt.title('Grad-CAM Heatmap for Bees') 
    
    plt.subplot(2, 2, 3)
    plt.imshow(img)
    #plt.title(f'Original image: {img_path.split('/')[-1]}')
    
    plt.subplot(2, 2, 4)
    plt.imshow(heatmap_ants, cmap='jet')
    plt.title('Grad-CAM Heatmap for Ants')
    
    plt.tight_layout()
    plt.show()

