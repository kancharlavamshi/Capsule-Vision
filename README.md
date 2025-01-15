![Capture](https://github.com/user-attachments/assets/b4e36bd6-345f-4d69-a735-c778adf4462a)

## CAPSULE VISION 2024 CHALLENGE
#    ** Multi-Class Abnormality Classification for Video Capsule Endoscopy**
The aim of this challenge, organized by MISAHUB(Medical Imaging and Signal Analysis Hub), is to advance the development, testing, and evaluation of AI models for the automatic classification of abnormalities in video capsule endoscopy (VCE) video frames. This initiative encourages the creation of vendor-independent and generalized AI-based models capable of accurately identifying and classifying various gastrointestinal abnormalities.

Challenge Details
Participants are tasked with developing an AI model to classify abnormalities captured in VCE video frames into one of the following 10 class labels:

1.Angioectasia
2.Bleeding
3.Erosion
4.Erythema
5.Foreign Body
6.Lymphangiectasia
7.Polyp
8.Ulcer
9.Worms
10.Normal

**Objectives**
Model Development: Create robust AI models that can effectively classify abnormalities based on the provided video frames.
Evaluation: Test and evaluate model performance using metrics such as accuracy, precision, recall, and F1 score.

<details>
<summary>Install</summary>
 Clone repo and install requirements.txt in a Python>=3.8.0 environment, including PyTorch>=1.8.

 ```
git clone https://github.com/kancharlavamshi/Capsule-Vision # clone
cd Capsule-Vision
pip install -r requirements.txt  # install
```
</details> 



<details>
<summary>Train</summary>
The commands below are used to train the model with specified configurations. The training process automatically handles data downloads and model initialization. Training times will vary depending on the model and the hardware used; expect different durations based on GPU capabilities. Use the largest --batch-size possible for optimal performance, or set --batch-size -1 for automatic batch sizing.
 
 ```
 python train.py --epochs 60 --batch-size 256 --save-model True --device cuda:0 --data_parallel True --parameters-print True --save-path ./output --model-name Model_ --validation-size 0.1 --L-r 0.0001 --Loss-func CrossEntropyLoss --Model-type efficient
```

</details> 


<details>
<summary>Validation</summary>
The following commands are used to validate the model and generate relevant metrics.
 
 ```
python validate.py --save-path ./output --Model-used efficient --con-matrix True --indiviual-accuracy-plt True --Save_Prediction True --metrics-report True --Train-xls-path ./training/training_data.xlsx --Val-xls-path ./validation/validation_data.xlsx 
```

</details> 


<details>
<summary>Test</summary>
The following commands are used to Test the model and generate Predictions(.csv/ xls file).
 
 ```
python prediction_test.py --save-path ./output --Model-used efficient --Print-prediction True
```

</details> 



 Confusion Matrix (Efficient+fusion)
![efficient_fusion_no_AttCM_](https://github.com/user-attachments/assets/93deb3b3-d6b3-481e-aa09-b060630b74d4)


## Please feel free to cite our paper if you use our code; this will help us continue our work and contribute more
## [Paper](https://arxiv.org/pdf/2410.19899)



 Individual  Accuracies (Efficient+fusion)
 ![efficient_fusion_no_AttInd_Cls_Acc](https://github.com/user-attachments/assets/94579dc9-2792-442b-9a3b-00afd0c7cc6d)

