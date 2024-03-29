# cGAN_vs_CNN_for_time_series_regression

## Imbalanced Time-Series Data Regression Using Conditional Generative Adversarial Networks

During the collection of time-series data, many reasons lead to imbalanced and incomplete datasets. Consequently, when training deep convolutional models on these datasets, the models suffer from overfitting and lack generalizability to unseen data. In this work, we investigated a new framework of Conditional Generative Adversarial Networks (cGANs) as a solution to improve the extrapolation and generalizability of the regression models in such datasets. We used an imbalanced synthetic dataset to show the advantage of a cGAN generator vs. a standalone CNN model. We will add a link to our paper that has more details about the synthesized data and the methods. 

### Prerequisites
To run this code, create a Python environment that contains the following libraries (numpy, pickle, os, random, math, tensorflow=2.5.0, sklearn, scipy=1.7.3, matplotlib), then run main.py. The synthesized data, trained models and results are saved in main_results folder. 
