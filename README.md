# Beard Classifier

## Overview

This repo is a test task for CV/ML engineer position that I've occasionally worked on

## Details

Dataset: 2001 images, 8 classes, imbalanced:
| |name			|Count	| Total, %  |
|-|---------------------|-------|-----------|
|0|chin_curtain 	|75   	|	 3.7|
|1|classic_long 	|186	|	 9.3|
|2|classic_short	|301	|    	15.0|
|3|goatee		|238	|    	11.9|
|4|goatee_with_moustache|301	|    	15.0|
|5|moustache		|298	|    	14.9|
|6|shaven		|301	|    	15.0|
|7|stubble		|301	|    	15.0|

In order to address issue with dataset imbalance I decided to use
pytorch's WeightedRandomSampler, that assigns class weight to each sample in
dataset. Less frequent class get higher weight. This results in equal probability of
each class being represented in during training. All images were scaled to 224x224
in order to perform computations faster.

General approach is image classification with CNN.

In order to rich high accuracy I decided to use transfer learning, more specifically:
ResNet34 trained on ImageNet dataset. Since beard classification is quite different
from what originally ResNet34 was trained on, I unfreeze later layers of CNN (the ones
that learnt high level representation). Also I replaced last fully-connected layer
with custom solution.

In order to prevent over-fitting I used several technique.
1. Data Augmentation. Random horizontal flip, rotation and horizontal translation.
2. Dropout, BatchNormalization layers.
3. Oversampling (described earlier) in order to prevent over-fitting to certain classes,
   that was controlled by per-class recall calculation.

Training: 45 epochs with SGD started at learning rate = 1e-3 decreased by factor of 10
each 15 epochs.

Training process you could find in file: 'training process.txt'

After each model were evaluated on whole dataset (evaluation process: 'evaluate process.txt')
Based on the results model with higher accuracy on whole dataset was selected (results below)

model name: 'beard_resnet34_epoch_30.pt'

| |name			|TP	|Total	|Accuracy, %|
|-|---------------------|-------|-------|-----------|
|0|chin_curtain		|70	|75	|93.33	    |
|1|classic_long		|180	|186	|96.77	    |
|2|classic_short	|288	|301	|95.68      |
|3|goatee		|231	|238	|97.06	    |
|4|goatee_with_moustache|282	|301	|93.69      |
|5|moustache		|294	|298	|98.66      |
|6|shaven		|293	|301	|97.34      |
|7|stubble		|284	|301	|94.35      |
|						    |
| |Average accuracy	|1922	|2001	|96.1%	    |

Further improvements could be done by exploiting deeper CNN, architectures, using ensembles of models, etc.
