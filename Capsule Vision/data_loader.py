
import os
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold


#classes = df_train.columns[2:].to_list()
#for i in range(0,len(classes)):
    #dd = df_train[df_train[classes[i]] == 1]
    #print(classes[i],len(dd))


def get_img_paths_labels(df):
    img_paths = []
    labels=[]
    classes = df.columns[2:].to_list()
    j = 0
    for i in range(0,len(classes)):    
        dd = df[df[classes[i]] == 1]
        #labels.extend(dd[classes[i]].to_list()*j)
        labels.extend([j] * len(dd[classes[i]].to_list()) )
        j += 1
        img_paths.extend(dd['image_path'].to_list())
    return img_paths,labels



# Function to balance datasets
def balance_data(image_paths, labels):
    label_counts = Counter(labels)
    min_samples = min(label_counts.values())  # Minimum samples across all classes

    balanced_image_paths = []
    balanced_labels = []

    for label in label_counts.keys():
        # Get indices of all samples with the current label
        label_indices = np.where(np.array(labels) == label)[0]
        
        # Randomly select min_samples indices
        selected_indices = np.random.choice(label_indices, min_samples, replace=False)
        
        # Append the selected samples to the balanced lists
        balanced_image_paths.extend(np.array(image_paths)[selected_indices])
        balanced_labels.extend(np.array(labels)[selected_indices])

    return balanced_image_paths, balanced_labels


class CustomDataset_clf(Dataset):
    def __init__(self, image_paths,labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        #image_paths = [path.replace("\\", "//") for path in image_paths]

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img_path = img_path.replace("\\", "/")
        image = Image.open("/home/neelamlab/Dataset/"+img_path).convert('RGB')
        org = self.normalize_image_to_01(image)        
        img_tensor = torch.tensor(org, dtype=torch.float32)
        img_tensor = img_tensor.permute(2, 0, 1)
        #if self.transform:
            #image = self.transform(image)
        labels = torch.tensor(label, dtype=torch.long)#.unsqueeze(0)
        
        return img_tensor, labels

    def normalize_image_to_01(self,image):
        image = np.array(image).astype(np.float32)
        return image / 255.0





def balance_data_2(image_paths, labels, desired_samples_per_class=None):
    label_counts = Counter(labels)
    
    if desired_samples_per_class is None:
        # Use the minimum number of samples across all classes
        min_samples = min(label_counts.values())
    else:
        min_samples = desired_samples_per_class
    
    balanced_image_paths = []
    balanced_labels = []

    for label in label_counts.keys():
        # Get indices of all samples with the current label
        label_indices = np.where(np.array(labels) == label)[0]
        
        # Check if the class has fewer samples than desired, if so take all
        available_samples = len(label_indices)
        if available_samples < min_samples:
            selected_indices = label_indices  # Take all available samples
        else:
            # Randomly select min_samples indices
            selected_indices = np.random.choice(label_indices, min_samples, replace=False)
        
        # Append the selected samples to the balanced lists
        balanced_image_paths.extend(np.array(image_paths)[selected_indices])
        balanced_labels.extend(np.array(labels)[selected_indices])

    return balanced_image_paths, balanced_labels

