This repository contains the synthetic dataset and real VSP dataset. 

This repository also allows to train 4 machine learning methods using the synthetic dataset : Multi-layer perceptron (MLP), one-dimension convolutional neural network (1D-CNN), support vector regressor (SVR), Extreme Gradient Boost (XGB).

#synthetic dataset
The synthetic dataset folder contains 300 simulation results of the 2D elastic wave equation. The features are extracted from peak and trough amplitude values of direct wave and reflected waves in time domain and peak amplitude values in frequency domain. The two lables are epsilon and delta respectively. These features of each simulation are reserved in Synthetic_Dataset.csv. 

#Real dataset
The real datset folder contains 22 real raw data acquired by VSP method with different offsets from 130 to 2000m. These data are splitted into upgoing waves and downgoing waves respectively and summarized in real_dataset.csv. 

#Models
The models fold contains four machine learning methods. 
