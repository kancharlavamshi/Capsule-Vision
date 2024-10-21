
import os
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import torch.nn.functional as F
from model import EfficientNetWithAttention,EfficientNetWithAttention_fusion,UNet,EfficientNetWithAttention_fusion_1


def conf_matrix(ground_truth_test, predictions_test,save_path,file_name):
    # class names from dataset
    class_names = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body',
                'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']

    # Confusion matrix calculation
    conf_matrix = confusion_matrix(ground_truth_test, predictions_test)

    # Plot confusion matrix using seaborn heatmap with class names
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix with Class Names')

    image_path = f"{save_path}/{file_name}CM_.png"
    # Save the plot
    plt.savefig(image_path)
    plt.show()

def indiviual_acc_plt(y_true, y_pred, save_path, file_name, class_names):
    # Get the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Compute individual class accuracies
    class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

    # Plotting the individual class accuracies
    plt.figure(figsize=(15, 6))
    bar_plot = sns.barplot(x=class_names[:len(class_accuracies)], y=class_accuracies, palette="viridis")
    plt.ylim(0, 1)
    plt.title('Individual Class Accuracies')
    plt.ylabel('Accuracy')
    plt.xlabel('Class')

    # Adding accuracy labels on top of each bar
    for p in bar_plot.patches:
        bar_plot.annotate(f'{p.get_height():.2f}', 
                          (p.get_x() + p.get_width() / 2., p.get_height()), 
                          ha='center', va='bottom', 
                          fontsize=12)

    # Save the plot as an image
    image_path = f"{save_path}/{file_name}Ind_Cls_Acc.png"
    plt.savefig(image_path)
    plt.show()


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

    
def test_inference(model,unet_model,Val_loader):
    Model_used = opt.Model_used
    save_path = opt.save_path
    Save_Prediction = opt.Save_Prediction

    from sklearn.metrics import accuracy_score
    test_accuracies=[]
    model.eval()
    # Evaluate on test set
    predictions_test = []
    ground_truth_test = []
    rows=[]
    with torch.no_grad():
        for images, labels in Val_loader:
            images, labels = images.to(device), labels.to(device)
            if Inference_model == "efficient":
                outputs = model(images)
            if Inference_model == "efficient_fusion":   
                _, unet_output = unet_model(images)
                outputs = model(images,unet_output)  
            if Inference_model == "efficient_fusion_no_Att":   
                _, unet_output = unet_model(images)
                outputs = model(images,unet_output)

            if Inference_model == "Unet_classifier":
               _, unet_output = unet_model(images)
               outputs = model(unet_output) 

            probabilities = F.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            predictions_test.extend(torch.round(outputs).cpu().numpy())
            ground_truth_test.extend(labels.cpu().numpy())
            rows.append(probabilities.cpu().numpy())

        class_names = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body','Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']            
        rows = np.concatenate(rows, axis=0)
        reshaped_rows = rows.reshape(len(rows), 10)
        new_rows_df = pd.DataFrame(rows, columns=class_names)
        if Save_Prediction == True:
            new_rows_df.to_csv(str(save_path)+str(Model_used)+'.csv')

        test_accuracy = accuracy_score(ground_truth_test, predictions_test)
    print('validation Accuracy:',test_accuracy)

    if confusion_matrix == True:
        file_name = Model_used
        conf_matrix(ground_truth_test, predictions_test,save_path,file_name)
    if indiviual_accuracy_plt == True:
        file_name = Model_used
        class_names =  ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body',
               'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']
        indiviual_acc_plt(ground_truth_test, predictions_test, save_path, file_name, class_names)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', type=str, default=os.getcwd())
    parser.add_argument('--Model-used', type=str, default="efficient")
    parser.add_argument('--confusion-matrix', type=str, default=False)
    parser.add_argument('--indiviual-accuracy-plt', type=str, default=False)        
    parser.add_argument('--Save_Prediction', type=str, default=False)

    parser.add_argument('--Train-xls-path', type=str, default='/home/neelamlab/Dataset/training/training_data.xlsx')
    parser.add_argument('--Val-xls-path', type=str, default='/home/neelamlab/Dataset/validation/validation_data.xlsx')

    opt = parser.parse_args()
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    Inference_model = opt.Model_used
    save_path = opt.save_path
    Model_used = opt.Model_used
    confusion_matrix = opt.confusion_matrix
    indiviual_accuracy_plt = opt.indiviual_accuracy_plt

    df_train = pd.read_excel(opt.Train_xls_path)
    df_val = pd.read_excel(opt.Val_xls_path)

    image_paths_train, labels_train = get_img_paths_labels(df_train)
    image_paths_val, labels_val = get_img_paths_labels(df_val)

        # Balance train and validation data
    image_paths_train_balanced, labels_train_balanced = balance_data(image_paths_train, labels_train)
    image_paths_val_balanced, labels_val_balanced = balance_data(image_paths_val, labels_val)


    # Instantiate the dataset (Balanced samples across each class)
    #train_dataset = CustomDataset_clf(image_paths_train_balanced, labels_train_balanced)
    ##Val_dataset = CustomDataset_clf(image_paths_val_balanced, labels_val_balanced)

    # Instantiate the dataset (Unbalanced)
    train_dataset = CustomDataset_clf(image_paths_train, labels_train)
    Val_dataset = CustomDataset_clf(image_paths_val, labels_val)

    batchsize = 16
    # Create the DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=0)
    Val_loader = DataLoader(Val_dataset, batch_size=batchsize, shuffle=True, num_workers=0)



    NUM_CLASSES = 10
    # Initialize model and data loaders




    unet_model = UNet(in_channels=3, out_channels=3)
    state_dict = torch.load("/data/data/Models/UNET_MASKED_7_million_500epochs_16oct.pth")
    if 'module.' in next(iter(state_dict.keys())):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    unet_model.load_state_dict(state_dict)
    unet_model = unet_model.to(device)

    if Inference_model == "efficient":
        model = EfficientNetWithAttention(num_classes=NUM_CLASSES)
        # Load the model state dictionary
        state_dict = torch.load("/home/neelamlab/Dataset/output/capsule_02oct_60_16_balanced_test.pth")
        # Handle 'DataParallel' case
        if 'module.' in next(iter(state_dict.keys())):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model = model.to(device)

    if Inference_model == "efficient_fusion":

        model = EfficientNetWithAttention_fusion(num_classes=NUM_CLASSES)
        # Load the model state dictionary
        state_dict = torch.load("/data/data/Models/Fusion_100_16.pth")
        # Handle 'DataParallel' case
        if 'module.' in next(iter(state_dict.keys())):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model = model.to(device)

    if Inference_model == "efficient_fusion_no_Att":

        model = EfficientNetWithAttention_fusion_1(num_classes=NUM_CLASSES)
        # Load the model state dictionary
        state_dict = torch.load("/data/data/Models/Fusion_no_attention150_256.pth")
        # Handle 'DataParallel' case
        if 'module.' in next(iter(state_dict.keys())):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model = model.to(device)

    if Inference_model == "Unet_classifier":
        model = CNNClassifier(num_classes=NUM_CLASSES)
        # Load the pretrained state dictionary
        state_dict = torch.load("/data/data/Models/Unet_classification_100_32.pth")

        # Handle 'DataParallel' case if necessary
        if 'module.' in next(iter(state_dict.keys())):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Load state dict into the model
        model.load_state_dict(state_dict)


    test_inference(model,unet_model,Val_loader)
 