# Follicle-Oocyte Ratio and its contribution for a Healthy Oocyte Maturation

## Overview
We propose a deep learning method to segment the areas of oocytes and follicles in a follicle image database. 
We utilize a pair of deep learning CNNs to segment the areas, while using some pre-processing and post-processing techniques to enhance the results. 
Then, we use the areas computed along with other data available for these follicles, to build a model for predicting a limited number of them to be used for maturation. 
The goal is to select the most likely promising follicles, while we are also studying the effect of the ratio oocyte-follicle to the final outcome. 
That way we can provide the experts with a tool for an automatic and safe follicle selection for human treatment in assisted reproductive technologies.

## Method

![Schema](figures/schema.png)

### ROI extraction
First we process the images to extract the region of interest (ROI) of the images using the 'ROI_pipeline.py' file. We resize both the images using the following principle: the region of interest if found by applying a local entropy kernel on all the image. Then, we threshold the obtained entropy mask to keep only the pixels where entropy is above the threshold. Using cv2 module 'findContours' method we extract the contours detected in the mask. Next, using a criterion based on the ratio between the area of the contour and its perimeter we are able to eliminate the artifacts contours that are present in the entropy mask by keeping only the round shaped contour corresponding to the cell. Thus the artifacts like the Petri dish border containing the cell are eliminated, and the algorithm succes rate found itself greatly increased.


### Follicle and oocyte segmentation
Next step is training a deep learning model in order to segment the different parts of the ovocytes (outter part and inner part). For that we chose to use the classical Unet neural network :
![Schema](figures/unet.png)

The training is made in two phases in the 'Attention_Unet_Inner_Part_Segmentation.py' and 'Attention_Unet_Outter_Part_Segmentation.py' files: first phase, we pretrain the Unet on 300 images from a dataset of a similar task dataset and second phase, we finetune the Unet on 65 images from our labelled oocyte dataset. The original loss function we used was the Dice Loss.

| Image from the pretraining dataset | Mask from the pretraining dataset |
|:-------:|:-------:|
| <img src="figures/eovo_530_t1.png" alt="Image 1" width="300px"> | <img src="figures/eovo_531_t1.png" alt="Image 2" width="300px"> |


### Ameliorations

We decided to modify the classical Unet model by adding  a spatial attention mechanism for better results like in the paper [arXiv:1804.03999]:


| Attentive Unet Architecture |
|:-------:|
| <img src="figures/attention_unet.png" alt="Image 4" width="300px"> |


This led to a better test dice coeficient, by helping the network to understand which part of the images were important for the segmentation. 
We first used the Dice Loss as loss function for the Unet, but using a "shape-aware" Dice Loss gave better results in terms of dice coefficient. This loss is defined as the product of the dice loss and a term equal to the euclidean distance between predicted and target masks contours. Therefore the Unet is forced to focus on the border of the segmentation zone, which is the crucial and most difficult part of the segmentation because it's where the artifacts are located. This approach also led to an increase of the dice coefficient.

### Predictive model

Next step, was to create a predictive model to predict if an oocyte will be able to go through maturation, using features such as the area of oocyte and follicle, the ratio between the two areas, disease of the oocyte donor, undergoing treatement of the donor and age of the donor. For this we used the XGBoost (eXtreme Gradient Boosting) model that we trained on a 600 vectors dataset where the areas and ratios where calculated with our predefined segmentation pipeline. 3 models were created, one for predicting binary maturation value after day 2 (1 if maturated 0 else), one for day 4 and one for day 6.

We used the SMOTE (Synthetic Minority Oversampling Technique) algorithm to get rid of the unbalanced classes issue.

## Probabilistic Predictions

Predicting the calasses (correct maturation /failed maturation) gives not so good results, thus a better approach is to obtain probabilistic predictions instead of binary classification. Using XGBoost python library we are able to make XGBoost produce probabilistic predictions and then to keep only the one that are above a threshold of certainty. By this way we are able to keep only a few predictions that we are sure about.

## Results

With the XGBoost library we can visualize feature importance for each of the three models and compute the achieved scores: 

| Feature Importance |
|:-------:|
| <p align="center"><img src="figures/features_importance.png" alt="Image 5" width="300px"></p> |


## Feature Importance

<p align="center">
  <img src="figures/features_importance.png" alt="Image 5" width="300px">
</p>








