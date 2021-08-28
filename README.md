# BrainTumour_Detection
A project that detects brain Tumors using layered Deep Learning Pipeline to perform Classification and Segmentation. Deep learning has been proven to be superior in detecting diseases from X-rays, MRI scans which could significantly improve the speed and accuracy of diagnosis. We have 3929 Brain scans along with their tumour location. 

# Approach
- Goal of image segmentation is to understand and extract information from images at the pixel level. 
- Image Segmentation can be used for object recognition and localization which offers tremendous value in many applications such as medical imaging and self driving cars. 
- The goal is to train a Neural Network to produce pixel-wise mask of the image.
- Mordern image segmentation techniques are based on deep learning approach which makes use of common architectures such as CNN, FCN.
- We will use ResUNet architecture to solve the task.
![Screenshot (101)](https://user-images.githubusercontent.com/70371572/131224250-27d0bf6d-7146-4287-8b7f-45400cafd58c.png)


## TASK 1. Import Libraries & Datasets
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import cv2
from skimage import io
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from IPython.display import display
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler, normalize
import os
import glob
import random
from google.colab import files #library to upload files to colab notebook
%matplotlib inline
```
Read data from data_mask.csv into brain_df

### Mask
- Goal of image segmentation is to understand image at pixel level. It associates each pixel with a certain class. The output produced by image segmentation model is called "mask" of the image.
- Masks can represented by pixel coordinates, we have flattened the array into 1-D array, we use the index to create the mask.
![Screenshot (102)](https://user-images.githubusercontent.com/70371572/131224305-1517418f-15bd-45dc-bf00-82d058532cf5.png)
```
brain_df.mask_path[1] # Path to the brain MRI image
brain_df.image_path[1] # Path to the segmentation mask
```

## Task 3. Perform Data Visualization
The mask distribution between all images is as follows,
![Screenshot (103)](https://user-images.githubusercontent.com/70371572/131223912-45985f1e-68b8-48eb-ba6c-8af261e07381.png)


Now we divide the images into 2 categories Brain MRI and Mask, see them in a grid layout
![Screenshot (112)](https://user-images.githubusercontent.com/70371572/131224051-abddc70b-48f9-4805-a487-9334358ad746.png)
![Screenshot (111)](https://user-images.githubusercontent.com/70371572/131224055-b1563786-183a-48b5-b2db-349ad4831703.png)

## TASK 4. Theory Behind CONVOLUTIONAL NEURAL NETWORKS AND RESNETS
- The first CNN layers are used to ectract high level general features.
- The last couple of layers are used to perform classification( on a specific task).
- Local fields scan the image for simple shapes such as edges/ lines.
- These edges are picked by subsequent layer to form more complex features.
- ![Screenshot (113)](https://user-images.githubusercontent.com/70371572/131224222-758b186b-0b79-4c9d-abd6-f5eca6ae5cd1.png)


### RESNET(Residual Network)

- As CNNs grow deeper, vanishing gradient tend to occur which negatively impacts performance.
- Residual Neural Network includes "skip connection" feature that enables training of 152 layers without vanishing Gradient Issue.
- This is done by identity mapping on top of CNN.
![Screenshot (114)](https://user-images.githubusercontent.com/70371572/131224218-31402017-ca6b-4746-97bc-2af6eccaef3e.png)

### Transfer Learning
- Transfer Learning is a machine learning technique in which a network that has been trained to perform a specific task is being reused(repurposed) as a starting point for another similar task.
- Transfer learning is widely used since starting from a pretrained models can dramatically reduce the time for computation as compared to if it started from scratch.
![Screenshot (115)](https://user-images.githubusercontent.com/70371572/131224453-7f05b9c9-e87c-4afd-a501-a04b73793271.png)

Ways to apply transfer learning:
- Strategy 1: 
  - Freeze the trained CNN network weights from the first layers.
  - Only train the newly added dense layers with random weights at initialization.
- Strategy 2: 
  - Initialize CNN network with the pre-trained weights.
  - Retain the entire CNN while setting learning rate to be very small this is critical to ensure that you do not aggressively change the trained weights.
- Transfer Learning approach:
  - Provides fast training progress, we don't have to start from scratch using randomly initialized weights.
  - We can use small training datasets to achieve excellent results.

Excellent Resource on transfer learning by Dipanjan Sarkar: https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a
Article by Jason Brownlee: https://machinelearningmastery.com/transfer-learning-for-deep-learning/

## TASK 5. Train a CLASSIFIER MODEL to detect if Tumor exists or not
- We drop the column patient id.
- We then split the data into training and test data.
- Now we create a imagegenerator which scales data from 0 to 1, makes a validation split of 0.15.
```
Found 2839 validated image filenames belonging to 2 classes.
Found 500 validated image filenames belonging to 2 classes.
Found 590 validated image filenames belonging to 2 classes.
```
- We get the ResNet50 base model by
```
basemodel = ResNet50(weights = 'imagenet', include_top = False, input_tensor = Input(shape=(256, 256, 3)))
basemodel.summary()
```
![Screenshot (116)](https://user-images.githubusercontent.com/70371572/131225544-60a14ba5-168e-4c28-9e91-aecb823381a3.png)
![Screenshot (117)](https://user-images.githubusercontent.com/70371572/131225547-6a107f81-4091-4207-8e65-6f716aee3012.png)
