## Implementation Code for XBL-D

This repository contains an implementation source code for a Distance-Aware Explanation based Learning (XBL-D), an IJCAI-23 submission paper. It contains scripts to extract COCO images, add confounding regions to them and generate a new dataset, tune a classifier, and refine a classifier using XBL-D.

To install all requirements: 
```
pip install -r requirements.txt 
```

For copyright issues, the decoy version of a subset of the MS-COCO dataset can not be shared here, and for research purposes it needs to be recreated using a Python script which is included in this repository, the decoy FashionMNSIT and decoy CIFAR-10 datasets can be downloaded from the following links:

- [Decoy FashionMNIST dataset](https://codeocean.com/capsule/7818629/tree/v1/data/fashion/decoy-fashion.npz): this dataset is created and shared as part of the paper [Schramowski et al. 'Making deep neural networks right for the right scientific reasons by interacting with their explanations.'](https://www.nature.com/articles/s42256-020-0212-3)
- [Decoy CIFAR-10 dataset](https://osf.io/w5f7y/?view_only=abb7f5f55bfc48fb8c891838f699c0d3): this dataset is shared anonymously as part of an IJCAI-2023 paper submission.

