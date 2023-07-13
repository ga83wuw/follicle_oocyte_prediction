# Follicle-Oocyte Ratio and its contribution for a Healthy Oocyte Maturation

## Overview
We propose a deep learning method to segment the areas of oocytes and follicles in a follicle image database. 
We utilize a pair of deep learning CNNs to segment the areas, while using some pre-processing and post-processing techniques to enhance the results. 
Then, we use the areas computed along with other data available for these follicles, to build a model for predicting a limited number of them to be used for maturation. 
The goal is to select the most likely promising follicles, while we are also studying the effect of the ratio oocyte-follicle to the final outcome. 
That way we can provide the experts with a tool for an automatic and safe follicle selection for human treatment in assisted reproductive technologies.

## Method

![Schema](figures/schema.png)

First we process the images to extract the region of interest (ROI) of the images using the 'ROI_pipeline.py' file. We resize both the images using the following principle: the region of interest if found by applying a local entropy kernel on all the image. Then, we theresold the obtained entropy mask to keep only the pixels where entropy is above the threshold. Using cv2 module 'findContours' method we extract the contours of the mask. Using a criterion based on the ratio between the area of the contour and its perimeter we are able to eliminate the artifacts that are present in the entropy mask by keeping only the round shaped contour corresponding to the cell. Thus the artifacts like the Petri dish border containing the cell are eliminated.
