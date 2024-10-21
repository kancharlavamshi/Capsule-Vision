
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
from models import EfficientNetWithAttention,EfficientNetWithAttention_fusion,UNet,EfficientNet_NoAttention_fusion




class CustomDataset_clf(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = self.find_jpg_files(image_paths)
        #image_paths = [path.replace("\\", "//") for path in image_paths]

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        org = self.normalize_image_to_01(image)        
        img_tensor = torch.tensor(org, dtype=torch.float32)
        img_tensor = img_tensor.permute(2, 0, 1)        
        return img_tensor, img_path


    def normalize_image_to_01(self,image):
        image = np.array(image).astype(np.float32)
        return image / 255.0

    def find_jpg_files(self,image_paths):
        # Use glob to find jpg files
        pattern = os.path.join(image_paths, '**', '*.jpg')
        jpg_files = glob.glob(pattern, recursive=True)
        return jpg_files        

    
def test_inference(model,unet_model,Val_loader):
    Model_used = opt.Model_used
    save_path = opt.save_path
    Save_Prediction = opt.Save_Prediction
    Print_prediction = opt.Print_prediction

    from sklearn.metrics import accuracy_score
    test_accuracies=[]
    model.eval()
    # Evaluate on test set
    predictions_test = []
    img_path = []
    rows=[]
    with torch.no_grad():
        for images, labels in Val_loader:
            images, labels = images.to(device), labels
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
            if Print_prediction.lower() == 'true':
                print('Predicted Class:',outputs)
            predictions_test.extend(torch.round(outputs).cpu().numpy())
            img_path.extend(labels)
            rows.append(probabilities.cpu().numpy())

        class_names = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body','Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']            
        rows = np.concatenate(rows, axis=0)
        reshaped_rows = rows.reshape(len(rows), 10)
        new_rows_df = pd.DataFrame(rows, columns=class_names)
        if Save_Prediction.lower() == 'true':
            new_rows_df.to_csv(str(save_path)+str(Model_used)+'.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', type=str, default=os.getcwd())
    parser.add_argument('--Model-used', type=str, default="efficient")      
    parser.add_argument('--Save_Prediction', type=str, default='False')
    parser.add_argument('--Test-img-path', type=str, default='/home/neelamlab/Dataset/Testing set/Images') 
    parser.add_argument('--Print-prediction', type=str, default='False') 

    opt = parser.parse_args()
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    Inference_model = opt.Model_used
    save_path = opt.save_path
    Model_used = opt.Model_used
    Test_img_path = opt.Test_img_path

    # Instantiate the dataset (Unbalanced)
    Test_dataset = CustomDataset_clf(Test_img_path)
    batchsize = 16
    # Create the DataLoader
    Test_loader = DataLoader(Test_dataset, batch_size=batchsize, shuffle=True, num_workers=0)



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

        model = EfficientNet_NoAttention_fusion(num_classes=NUM_CLASSES)
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


    test_inference(model,unet_model,Test_loader)
 