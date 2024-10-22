import argparse
import os
import torch
from torch.utils.data import random_split, DataLoader
from data_loader import CustomDataset_clf,balance_data,get_img_paths_labels,balance_data_2
from torch import optim
from torchvision import models
import pandas as pd
from torch.utils.data import Dataset, DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from torch import nn, optim
import torch.nn as nn
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
import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
###____________________________________#########################
import random
from models import EfficientNetWithAttention,EfficientNetWithAttention_fusion,UNet,EfficientNet_NoAttention_fusion,CNNClassifier

torch.manual_seed(98)
random.seed(98)
np.random.seed(98)



##  check for GPU ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Initialize UNet model
unet_model = UNet(in_channels=3, out_channels=3)
# Load the pretrained state dictionary
state_dict = torch.load("/data/data/Models/UNET_MASKED_7_million_500epochs_16oct.pth")
# Handle 'DataParallel' case if necessary
if 'module.' in next(iter(state_dict.keys())):
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
# Load state dict into the model
unet_model.load_state_dict(state_dict)

class FocalLoss_1(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss_1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # Compute cross entropy loss
        pt = torch.exp(-ce_loss)  # pt is the probability of the correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  # Apply Focal Loss formula

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train_run(model,unet_model, LEARNING_RATE, EPOCHS):
    parameters_print = opt.parameters_print
    data_parallel = opt.data_parallel
    save_path = opt.save_path
    model_name = opt.model_name
    loss_function = opt.Loss_func
    learning_rate = opt.L_r
    Model_type = opt.Model_type


    criterion_1 = FocalLoss_1(alpha=0.25, gamma=2)
    optimizer_1 = optim.Adam(model.parameters(), lr=learning_rate)

    criterion_2 = nn.CrossEntropyLoss()
    optimizer_2 = optim.Adam(model.parameters(), lr=learning_rate)


    # Data Parallel
    if data_parallel.lower() == 'true' and device:
        model = nn.DataParallel(model)
        unet_model = nn.DataParallel(unet_model)

    else:
        if data_parallel.lower() == 'true':
            print("Data parallelism requires a CUDA device. Please specify a CUDA device.")

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count the number of parameters
    if parameters_print.lower() == 'true':
        num_params = count_parameters(model)
        print("Number of parameters in the model:", num_params)



    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    best_test_acc = float('-inf')

        
    for epoch in range(EPOCHS):
        predictions = []
        ground_truth = []
        model.train()  # Set the model to training mode
        running_loss = 0.0

        # Choose the criterion and optimizer based on the epoch
        if loss_function == 'FocalLoss':
            criterion = criterion_1
            optimizer = optimizer_1
        else:
            criterion = criterion_2
            optimizer = optimizer_2

        
        # Training loop
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Perform the forward pass through the UNet model
            unet_model.eval()  # UNet is used for feature extraction
            with torch.no_grad():  # Disable gradients for inference
                _, unet_output = unet_model(images)  # Forward pass through UNet
            # Proceed with the forward pass for your model
            optimizer.zero_grad()


            if Model_type == "efficient":
                outputs = model(images)
            if Model_type == "efficient_Att_Fusion":
                # Pass images and the bottleneck output to the model
                outputs = model(images, unet_output)
            if Model_type == "EfficientNet_NoAtt_Fusion":
                # Pass images and the bottleneck output to the model
                outputs = model(images, unet_output)    
            if Model_type == "CNN_Classifier":
                outputs = model(unet_output)


            # Compute the loss
            train_loss = criterion(outputs, labels)

            # Backpropagation
            train_loss.backward()

            # Optimizer step
            optimizer.step()

            # Calculate running loss
            running_loss += train_loss.item() * images.size(0)

            # Convert predictions to binary class
            outputs = torch.argmax(outputs, dim=1)
            predictions.extend(torch.round(outputs).detach().cpu().numpy())
            ground_truth.extend(labels.detach().cpu().numpy())

        # Calculate training accuracy
        train_accuracy = accuracy_score(ground_truth, predictions)
        train_accuracies.append(train_accuracy)

        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader)

        # Remove the hook after the training loop


        # Validation/Test Loop
        model.eval()
        predictions_test = []
        ground_truth_test = []

        unet_model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                # Perform forward pass through the UNet
                _, unet_output = unet_model(images)  # Forward pass through UNet

                if Model_type == "efficient":
                    outputs = model(images)
                if Model_type == "efficient_Att_Fusion" or "EfficientNet_NoAtt_Fusion":
                    # Pass images and the bottleneck output to the model
                    outputs = model(images, unet_output)
                if Model_type == "CNN_Classifier":
                    outputs = model(unet_output)

                # Convert predictions to binary class
                outputs = torch.argmax(outputs, dim=1)
                predictions_test.extend(torch.round(outputs).cpu().numpy())
                ground_truth_test.extend(labels.cpu().numpy())

            # Calculate test accuracy
            test_accuracy = accuracy_score(ground_truth_test, predictions_test)
            test_accuracies.append(test_accuracy)


        print(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss:.4f}, '
            f'Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}')

        # Save Model
        if opt.save_model.lower() == 'true':
            if best_test_acc < test_accuracy:
                best_test_acc = test_accuracy
                torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}{EPOCHS}_{opt.batch_size}.pth"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=256, help='total batch size for all GPUs')
    parser.add_argument('--save-model', type=str, default='False', help='save the model')
    parser.add_argument('--device', type=str, default='True', help='Device to use for training (e.g., cpu, cuda:0)')
    parser.add_argument('--data_parallel', type=str, default='False', help='Data Parallel (Multiple GPUs)')
    parser.add_argument('--parameters-print', type=str, default='False', help='Print No.of Model Parameters')
    parser.add_argument('--save-path', default=os.getcwd(), help='Path to save the output (default: current directory)')
    parser.add_argument('--model-name', default='Model_', help='Model name for saving')
    parser.add_argument('--validation-size', type=float, default=0.1, help='Validation Size')
    parser.add_argument('--L-r', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--Loss-func', type=str, default='CrossEntropyLoss', help='Loss function')
    parser.add_argument('--Model-type', type=str, default="efficient")


    opt = parser.parse_args()
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    NUM_CLASSES = 10
    # Initialize model and data loaders
    Model_type = opt.Model_type
    if Model_type == "efficient":
       model = EfficientNetWithAttention(num_classes=NUM_CLASSES)
    if Model_type == "efficient_Att_Fusion":
       model = EfficientNetWithAttention_fusion(num_classes=NUM_CLASSES)

    if Model_type == "EfficientNet_NoAtt_Fusion":
       model = EfficientNet_NoAttention_fusion(num_classes=NUM_CLASSES)

    if Model_type == "CNN_Classifier":
       model = CNNClassifier(num_classes=NUM_CLASSES)      



    Train_model = model.to(device)
    unet_model = unet_model.to(device)
    #dummy_input = torch.randn(10, 3, 224, 223).to(device)
    #output = Train_model(dummy_input,dummy_input)
    #print('Output shape:', output.shape)

    import pandas as pd
    file_path_train = '/home/neelamlab/Dataset/training/training_data.xlsx'
    file_path_val = '/home/neelamlab/Dataset/validation/validation_data.xlsx'

    df_train = pd.read_excel(file_path_train)
    df_val = pd.read_excel(file_path_val)
    
    image_paths_train, labels_train = get_img_paths_labels(df_train)
    image_paths_val, labels_val = get_img_paths_labels(df_val)

        # Balance train and validation data
    image_paths_train_balanced, labels_train_balanced = balance_data(image_paths_train, labels_train)
    image_paths_val_balanced, labels_val_balanced = balance_data(image_paths_val, labels_val)


    
    from torch.utils.data import Dataset, DataLoader


    # Instantiate the dataset (Balanced samples across each class)
    #train_dataset = CustomDataset_clf(image_paths_train_balanced, labels_train_balanced)
    #test_dataset = CustomDataset_clf(image_paths_val_balanced, labels_val_balanced)

    # Instantiate the dataset (Unbalanced)
    train_dataset = CustomDataset_clf(image_paths_train, labels_train)
    test_dataset = CustomDataset_clf(image_paths_val, labels_val)



    # Create the DataLoader
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)

    train_run(Train_model, unet_model,opt.L_r, opt.epochs)