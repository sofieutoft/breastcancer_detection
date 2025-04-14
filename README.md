# **Liver and Liver Tumor Segmentation: A Comparison of Model Efficacy**

## Description
This project aims to benchmark various image segmentation machine learning models. In order to do so, we will make use of the Liver Tumor Segmentation (LiTS) dataset. One motivation for doing so is that the early and accurate identification of liver tumors is crucial for timely medical intervention and improved patient outcomes. This dataset is available on Kaggle: https://www.kaggle.com/datasets/andrewmvd/lits-png. This dataset consists of preprocessed CT scan images of the liver with diverse types of both primary and secondary tumors all taken with various lesion-to-background levels.

We suspect that our goal will be particularly challenging, given the heterogeneous and diffuse shape of liver tumors. Still, our aim is to provide a benchmark for various model architectures, ranging from classical machine learning techniques to deep learning algorithms. Here is a tentative list of models we will compare:
1. K-Means Clustering 
2. Thresholding
3. U-Net

The main metric used to assess the performance of our models will be Intersection over Union (IoU) which measures the overlap between the predicted segmentation and the ground truth. The higher the IoU value, the more faithful our segmentation is to the ground truth. However we are not ruling out using other performance metrics such as the dice coefficient or overall pixel accuracy (however these all get at the same idea of rationing the overlap of correctly segmented pixels). To make this project as holistic as possible, we will also consider the effect different loss functions have on model performance. 
