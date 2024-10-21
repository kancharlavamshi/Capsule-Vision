import os
import pandas as pd
import torch
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn, optim
import numpy as np
from torchvision import transforms
from torchsummary import summary
from torch.utils.data import random_split
from sklearn.metrics import mean_absolute_error
import seaborn as sns
from math import ceil
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from PIL import Image
import glob
from tqdm import tqdm
###____________________________________#########################
import random
torch.manual_seed(98)
random.seed(98)
np.random.seed(98)

from models import EfficientNetWithAttention_fusion,UNet



##  check for GPU ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def find_jpg_files(parent_folder):
    # Use glob to find jpg files
    pattern = os.path.join(parent_folder, '**', '*.jpg')
    jpg_files = glob.glob(pattern, recursive=True)
    return jpg_files

parent_folder = '/home/neelamlab/Dataset/training'
jpg_files = find_jpg_files(parent_folder)
for file in jpg_files:
    #print(file)
    img = Image.open(file)
    break


def divide_image_into_patches(image, patch_size, num_patches):
    patches = []
    img_width, img_height = image.size
    
    for i in range(num_patches):
        row = i // int(np.sqrt(num_patches))
        col = i % int(np.sqrt(num_patches))
        
        left = col * patch_size
        upper = row * patch_size
        right = left + patch_size
        lower = upper + patch_size
        
        patch = image.crop((left, upper, right, lower))
        patches.append(patch)
    
    return patches

def mask_patches(patches, mask_fraction):
    num_patches = len(patches)
    num_to_mask = int(num_patches * mask_fraction)
    indices = list(range(num_patches))
    random.shuffle(indices)
    
    # Indices to move to the end
    indices_to_move = {0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 23, 24, 31, 32, 39, 40, 47, 48, 55, 56, 57, 58, 59, 60, 61, 62, 63}    
    # Separate the indices to move and the rest
    to_move = [index for index in indices if index in indices_to_move]
    rest = [index for index in indices if index not in indices_to_move]   
    # Create the new list
    new_indices = rest + to_move
    #print('after indices', new_indices)
    mask_indices = set(new_indices[:num_to_mask])

    
    #mask_indices = set(indices[:num_to_mask])
    
    masked_patches = []
    for i, patch in enumerate(patches):
        if i in mask_indices:
            # Create a mask (e.g., black patch)
            mask = Image.new('RGB', patch.size, (0, 0, 0))
            masked_patches.append(mask)
        else:
            masked_patches.append(patch)
    
    return masked_patches

def combine_patches_into_image(patches, patch_size, num_patches_per_row):
    num_patches = len(patches)
    img_width = patch_size * num_patches_per_row
    img_height = patch_size * (num_patches // num_patches_per_row)
    
    combined_image = Image.new('RGB', (img_width, img_height))
    
    for i, patch in enumerate(patches):
        row = i // num_patches_per_row
        col = i % num_patches_per_row
        
        left = col * patch_size
        upper = row * patch_size
        combined_image.paste(patch, (left, upper))
    
    return combined_image


# Parameters
patch_size = 224 // 20  # Each patch size (224x224 divided into 8x8 patches)
num_patches_per_row = 20  # Number of patches per row
num_patches = num_patches_per_row * num_patches_per_row  # Total patches
mask_fraction = 0.25  # Fraction of patches to mask

# Divide image into patches
patches = divide_image_into_patches(img, patch_size, num_patches)

# Mask patches
masked_patches = mask_patches(patches, mask_fraction)

# Combine patches into a single image
combined_image = combine_patches_into_image(masked_patches, patch_size, num_patches_per_row)






import cv2 
def add_gaussian_noise(image, mean=0, std=0.54):
    """
    Adds Gaussian noise to an image.

    Parameters:
        image (numpy array): The input image.
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.

    Returns:
        noisy_image (numpy array): The image with added Gaussian noise.
    """
    image = np.array(image.convert('RGB'))
    # Generate Gaussian noise
    gaussian_noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    
    # Add noise to the image
    noisy_image = cv2.add(image, gaussian_noise)
    
    return noisy_image





class CustomDataset_gaussian():
    def __init__(self, data_dir, transform = None):  
        self.file_paths = self.get_file_paths_registered(data_dir)
        self.transform = transform       
    def __len__(self):
        return len(self.file_paths)    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = Image.open(file_path)
        combined_image = add_gaussian_noise(image)

        org = self.normalize_image_to_01(image)        
        org_tensor = torch.tensor(org, dtype=torch.float32)
        org_tensor = org_tensor.permute(2, 0, 1).to(device)
        
        masked = self.normalize_image_to_01(combined_image)        
        masked_tensor = torch.tensor(masked, dtype=torch.float32).to(device)
        masked_tensor = masked_tensor.permute(2, 0, 1).to(device)

        
        return masked_tensor,org_tensor                     
    
    def get_file_paths_registered(self,path):
        def find_jpg_files(parent_folder):
            # Use glob to find jpg files
            pattern = os.path.join(parent_folder, '**', '*.jpg')
            jpg_files = glob.glob(pattern, recursive=True)
            return jpg_files
        jpg_files = find_jpg_files(path)
                    
        return jpg_files 
    def preprocess_data_2(self, data):
        data = np.array(data).astype(np.float32)
        mean = data.mean()
        std = data.std()
        normalized_data = (data - mean) / std
        return normalized_data
    def normalize_image_to_01(self,image):
        image = np.array(image).astype(np.float32)
        return image / 255.0
    def normalize_image(self, image):
        image = np.array(image).astype(np.float32)
        image = image / 255.0       
        mean = np.array([0.5501522, 0.34230372, 0.1798497])
        std = np.array([0.19262312, 0.13819472, 0.09009915])        
        image = (image - mean) / std        
        return image 

    def add_gaussian_noise(self,image, mean=0, std=0.54):
        """
        Adds Gaussian noise to an image.

        Parameters:
            image (numpy array): The input image.
            mean (float): Mean of the Gaussian noise.
            std (float): Standard deviation of the Gaussian noise.

        Returns:
            noisy_image (numpy array): The image with added Gaussian noise.
        """
        image = np.array(image.convert('RGB'))
        # Generate Gaussian noise
        gaussian_noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
        
        # Add noise to the image
        noisy_image = cv2.add(image, gaussian_noise)
        
        return noisy_image            















class CustomDataset():
    def __init__(self, data_dir, transform = None):  
        self.file_paths = self.get_file_paths_registered(data_dir)
        self.transform = transform       
    def __len__(self):
        return len(self.file_paths)    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = Image.open(file_path)

        # Parameters
        patch_size = 224 // 28  
        num_patches_per_row = 28  
        num_patches = num_patches_per_row * num_patches_per_row  # Total patches
        mask_fraction = 0.25  # Fraction of patches to mask
        # Divide image into patches
        patches = divide_image_into_patches(image, patch_size, num_patches)
        # Mask patches
        masked_patches = mask_patches(patches, mask_fraction)
        # Combine patches into a single image
        combined_image = combine_patches_into_image(masked_patches, patch_size, num_patches_per_row)
        
        org = self.normalize_image_to_01(image)        
        org_tensor = torch.tensor(org, dtype=torch.float32)
        org_tensor = org_tensor.permute(2, 0, 1).to(device)
        
        masked = self.normalize_image_to_01(combined_image)        
        masked_tensor = torch.tensor(masked, dtype=torch.float32).to(device)
        masked_tensor = masked_tensor.permute(2, 0, 1).to(device)

        
        return masked_tensor,org_tensor                     
    
    def get_file_paths_registered(self,path):
        def find_jpg_files(parent_folder):
            # Use glob to find jpg files
            pattern = os.path.join(parent_folder, '**', '*.jpg')
            jpg_files = glob.glob(pattern, recursive=True)
            return jpg_files
        jpg_files = find_jpg_files(path)
                    
        return jpg_files 
    def preprocess_data_2(self, data):
        data = np.array(data).astype(np.float32)
        mean = data.mean()
        std = data.std()
        normalized_data = (data - mean) / std
        return normalized_data
    def normalize_image_to_01(self,image):
        image = np.array(image).astype(np.float32)
        return image / 255.0
    def normalize_image(self, image):
        image = np.array(image).astype(np.float32)
        image = image / 255.0       
        mean = np.array([0.5501522, 0.34230372, 0.1798497])
        std = np.array([0.19262312, 0.13819472, 0.09009915])        
        image = (image - mean) / std        
        return image    
    """
    def divide_image_into_patches(image, patch_size, num_patches):
        patches = []
        img_width, img_height = image.size
        
        for i in range(num_patches):
            row = i // int(np.sqrt(num_patches))
            col = i % int(np.sqrt(num_patches))
            
            left = col * patch_size
            upper = row * patch_size
            right = left + patch_size
            lower = upper + patch_size
            
            patch = image.crop((left, upper, right, lower))
            patches.append(patch)
        
        return patches

    def mask_patches(patches, mask_fraction):
        num_patches = len(patches)
        num_to_mask = int(num_patches * mask_fraction)
        
        indices = list(range(num_patches))
        random.shuffle(indices)
        mask_indices = set(indices[:num_to_mask])
        
        masked_patches = []
        for i, patch in enumerate(patches):
            if i in mask_indices:
                # Create a mask (e.g., black patch)
                mask = Image.new('RGB', patch.size, (0, 0, 0))
                masked_patches.append(mask)
            else:
                masked_patches.append(patch)
        
        return masked_patches


    
    def combine_patches_into_image(patches, patch_size, num_patches_per_row):
        num_patches = len(patches)
        img_width = patch_size * num_patches_per_row
        img_height = patch_size * (num_patches // num_patches_per_row)
        
        combined_image = Image.new('RGB', (img_width, img_height))
        
        for i, patch in enumerate(patches):
            row = i // num_patches_per_row
            col = i % num_patches_per_row
            
            left = col * patch_size
            upper = row * patch_size
            combined_image.paste(patch, (left, upper))
        
        return combined_image
    """

    def divide_image_into_patches(image, patch_size, num_patches):
        patches = []
        img_width, img_height = image.size
        
        for i in range(num_patches):
            row = i // int(np.sqrt(num_patches))
            col = i % int(np.sqrt(num_patches))
            
            left = col * patch_size
            upper = row * patch_size
            right = left + patch_size
            lower = upper + patch_size
            
            patch = image.crop((left, upper, right, lower))
            patches.append(patch)
        
        return patches

    def mask_patches(patches, mask_fraction):
        num_patches = len(patches)
        num_to_mask = int(num_patches * mask_fraction)
        indices = list(range(num_patches))
        random.shuffle(indices)
        
        # Indices to move to the end
        indices_to_move = {0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 23, 24, 31, 32, 39, 40, 47, 48, 55, 56, 57, 58, 59, 60, 61, 62, 63}    
        # Separate the indices to move and the rest
        to_move = [index for index in indices if index in indices_to_move]
        rest = [index for index in indices if index not in indices_to_move]   
        # Create the new list
        new_indices = rest + to_move
        #print('after indices', new_indices)
        mask_indices = set(new_indices[:num_to_mask])

        
        #mask_indices = set(indices[:num_to_mask])
        
        masked_patches = []
        for i, patch in enumerate(patches):
            if i in mask_indices:
                # Create a mask (e.g., black patch)
                mask = Image.new('RGB', patch.size, (0, 0, 0))
                masked_patches.append(mask)
            else:
                masked_patches.append(patch)
        
        return masked_patches

    def combine_patches_into_image(patches, patch_size, num_patches_per_row):
        num_patches = len(patches)
        img_width = patch_size * num_patches_per_row
        img_height = patch_size * (num_patches // num_patches_per_row)
        
        combined_image = Image.new('RGB', (img_width, img_height))
        
        for i, patch in enumerate(patches):
            row = i // num_patches_per_row
            col = i % num_patches_per_row
            
            left = col * patch_size
            upper = row * patch_size
            combined_image.paste(patch, (left, upper))
        
        return combined_image






















model = UNet(in_channels=3, out_channels=3)
# Create a dummy input tensor with shape (batch_size, channels, height, width)
input_tensor = torch.randn(1, 3, 224, 224)
# Forward pass
output,bottleneck = model(input_tensor)

# Print the output shape
print("output",output.shape,"bottleneck",bottleneck.shape) 

#summary(model,input_size=(1,3,224,224))

data_dir_train = "/home/neelamlab/Dataset/training"
data_dir_test = "/home/neelamlab/Dataset/validation"

batch_size = 256 
##task = 'gaussian'
task = 'patch'
num_epochs = 200




if task == 'patch':

    train_dataset = CustomDataset(data_dir_train)#,transform=transform)
    test_dataset = CustomDataset(data_dir_test)#,transform=transform)

else:
    train_dataset = CustomDataset_gaussian(data_dir_train)#,transform=transform)
    test_dataset = CustomDataset_gaussian(data_dir_test)#,transform=transform)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

criterion = nn.MSELoss()

optimizer_adam = optim.Adam(model.parameters(), lr=0.0001)##,weight_decay=0.0001)

optimizer = optimizer_adam

import torch
import torch.nn as nn
import torch.optim as optim


# Initialize model, optimizer, and criterion
model = UNet(in_channels=3, out_channels=3)#.to(device)
model = nn.DataParallel(model)  # Wrap model for data parallelism
model  = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.MSELoss()

train_l = []  
best_test_loss = float('inf')

for epoch in range(num_epochs):
    total_loss_Train = 0.0
    total_loss_Test = 0.0
    count_train = 0
    count_test = 0
    train_ss = 0
    train_pp = 0
    test_ss = 0
    test_pp = 0

    model.train()
    for batch_data, batch_labels in train_loader:
        optimizer.zero_grad()

        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)

        outputs,_ = model(batch_data)
        
        loss_train = criterion(outputs, batch_labels)
        total_loss_Train += loss_train.item()
        count_train += 1
        loss_train.backward()
        optimizer.step()
    
    # Calculate training loss
    train_loss_v = total_loss_Train / count_train
   

    # Validation
    model.eval()
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            outputs,_ = model(batch_data)
            loss_test = criterion(outputs, batch_labels)
            total_loss_Test += loss_test.item()
            count_test += 1
    
    # Calculate validation loss
    if total_loss_Test == 0:
        test_loss_v = count_test
    else:    
        test_loss_v = total_loss_Test / count_test


    print(f"Epoch {epoch+1}, Training loss: {train_loss_v}, Validation loss: {test_loss_v}")
    
    # Save model based on best validation accuracy
    if test_loss_v < best_test_loss:
        best_test_loss = test_loss_v
        torch.save(model.state_dict(), "/data/data/Models/UNET_MASKED_7_million_200epochs_18oct_patch.pth")



import torch
import matplotlib.pyplot as plt
import numpy as np
model = model.to(device)
for batch_data, batch_labels in test_loader:
    batch_data = batch_data.to(device)
    #batch_labels = batch_labels.to(device)
    outputs,_ = model(batch_data)
    # Example tensors (replace with your actual tensors)
    batch_data = batch_data.cuda()  # Replace with your actual batch_data tensor
    outputs = outputs.cuda()     # Replace with your actual outputs tensor

    # Move tensors to CPU and detach from the computation graph
    batch_data_cpu = batch_data.cpu().detach()
    outputs_cpu = outputs.cpu().detach()

    # Convert tensors to NumPy arrays and permute dimensions
    batch_data_np = batch_data_cpu.permute(0, 2, 3, 1).numpy()  # Shape: [7, 224, 224, 3]
    outputs_np = outputs_cpu.permute(0, 2, 3, 1).numpy()      # Shape: [7, 224, 224, 3]

    # Plot each image in a grid
    fig, axes = plt.subplots(2, 7, figsize=(20, 10))  # 2 rows, 7 columns
    print('batch_data_np',batch_data_np.shape)
    print('batch_data_cpu',batch_data_cpu.shape)
    for i in range(7):
        # Batch data images
        image_batch = batch_data_np[i]
        image_batch = (image_batch - image_batch.min()) / (image_batch.max() - image_batch.min())  # Normalize to [0, 1]
        
        ax_batch = axes[0, i]
        ax_batch.imshow(image_batch)
        ax_batch.axis('off')
        ax_batch.set_title(f'Batch {i+1}')
        
        # Output images
        image_output = outputs_np[i]
        image_output = (image_output - image_output.min()) / (image_output.max() - image_output.min())  # Normalize to [0, 1]
        
        ax_output = axes[1, i]
        ax_output.imshow(image_output)
        ax_output.axis('off')
        ax_output.set_title(f'Output {i+1}')

    plt.tight_layout()
    #plt.show()
    plt.savefig('/data/data/Models/UNET_MASKED_7_million_200epochs_18oct_patch.png',dpi=400)
    break