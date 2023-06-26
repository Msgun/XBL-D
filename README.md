## Implementation Code for XBL-D

This repository contains an implementation source code for Distance-Aware Explanation based Learning (XBL-D) paper. It contains scripts to extract COCO images, add confounding regions to them and generate a new dataset, tune a classifier, and refine a classifier using XBL-D.

To install all requirements: 
```
pip install -r requirements.txt 
```

For copyright issues, the decoy version of a subset of the MS-COCO dataset can not be shared here, and for research purposes it needs to be recreated using a Python script which is included in this repository. The decoy FashionMNSIT and decoy CIFAR-10 datasets can be downloaded from the following links:

<ul>
  <li>Decoy FashionMNIST dataset: this dataset is created and shared as part of the paper Schramowski et al. [2020] 'Making deep neural networks right for the right scientific reasons by interacting with their explanations.' It can be downloaded from the following link: https://codeocean.com/capsule/7818629/tree/v1/data/fashion/decoy-fashion.npz </li>
  <li>Decoy CIFAR-10 dataset: this dataset is shared anonymously. It can be downloaded from the following link: https://osf.io/w5f7y/?view_only=abb7f5f55bfc48fb8c891838f699c0d3 </li>
</ul> 

<!---
- [Decoy FashionMNIST dataset](https://codeocean.com/capsule/7818629/tree/v1/data/fashion/decoy-fashion.npz): this dataset is created and shared as part of the paper [Schramowski et al. 'Making deep neural networks right for the right scientific reasons by interacting with their explanations.'](https://www.nature.com/articles/s42256-020-0212-3)
- [Decoy CIFAR-10 dataset](https://osf.io/w5f7y/?view_only=abb7f5f55bfc48fb8c891838f699c0d3): this dataset is anonymously shared as part of an IJCAI-2023 paper submission from the owners of this code repository.
-->

<!---
coco_data_extraction took 17 minutes
coco_decoy_generate took 2 minutes
tuning took 9 minutes

-->

# Instructions


## Downloading MS COCO JSON

Download the 'instances_train2014.json' from [this link](https://cocodataset.org/#download) and place it in the same directory as the scripts in this repository. It is found inside the '2014 Train/Val annotations.'

## Extraction of Images and Annotations from MS COCO

To extract training and testing images and object annotations for the Train and Zebra categories from the MS COCO database using the 'instances_train2014.json' file:
```
python coco_data_extraction.py
```
This will create a directory named 'MSCOCO' with two sub-directories named 'images' and 'annotations' and populate them with images and object annotations of the Train and Zebra categories.
## Generating a Decoy Version of the Train and Zebra Categories of MS COCO

To add confounding regions to the extracted images and generate masks of the confounders:
```
python coco_decoy_generate.py
```
This will create two sub-directories inside 'MSCOCO' named 'confounded' and 'confounded_mask' and populate them with confounded versions of the images and mask annotations of the confounders.
## Hyperparameter search and pretraining

To search for the optimal hyperparameters: (we consider the hyperparameters: number and size of convolutional layers, number of pooling layers, number and size of fully connected layers, and learning rate for each datasets separately.)
```
python tuning_classifier.py
```

This will create a directory named 'models_coco' that contains tuned models. In addition, it will select the best performing model architecture and train it using classification loss. At then end, the trained model will be saved as 'coco.h5'

## Training

To start training a model using XBL-D:
```
python xbl-d_train.py
```
This will load the pretrained coco.h5 from above and refine it using XBL-D.