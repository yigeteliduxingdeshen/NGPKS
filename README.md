# NGPKS
To better predict miRNA-disease association, we propose a Non-linear Gaussian Profile Kernel Similarity method. Prediction performance is improved by combining miRNA-miRNA non-linear Gaussian profile kernel functional similarity matrix, diseasease nonlinear Gaussian profile kernel semantic similarity matrix, and miRNA-disease association matrix to extract potential features between data. This repository contains codes and datas for HyperVR model.
# Data description

| File name           | Description                                                                                   |
|---------------------|-----------------------------------------------------------------------------------------------|
| m-d.csv             | miRNA-disease association matrix                                                              |
| m-m.csv             | miRNA-miRNA functional similarity matrix                                                      |
| M_GSM.txt           | miRNA Non-linear Gaussian Profile kernel similarity matrix                                    |
| d-d.csv             | Semantic similarity matrix for disease-disease                                                |
| D_GSM.txt           | disease Non-linear Gaussian Profile kernel similarity matrix                                  | 
| miRNA name.csv      | Names of all miRNAs in the m-m file                                                           | 
| disease name.csv    | Names of all disease in the d-d file                                                          |


NGPKS:  Non-linear Gaussian profile kernel Similarity and Convolutional Networks for miRNA-disease association prediction. 

# Requirements
HyperVR is tested to work under:

Python 3.9

Pytorch 1.10.1+cu113

numpy 1.22.3

sklearn 1.1.1

# Quick start
To reproduce our results:
1. Unzip NGPKS.zip in ./NGPKS.
2. Run calculate_Gaussian.py to generate the corresponding D_GSM.txt and M_GSM.txt files
3. Run train.py to generate the training model file
4. Run validation.py to import the corresponding model and then perform prediction validation




