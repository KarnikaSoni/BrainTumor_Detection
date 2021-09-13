# BrainTumor_Detection
A project that detects brain Tumors using layered Deep Learning Pipeline to perform Classification and Segmentation. Deep learning has been proven to be superior in detecting diseases from X-rays, MRI scans which could significantly improve the speed and accuracy of diagnosis. We have 3929 Brain scans along with their tumor location. 

# Approach
- Goal of image segmentation is to understand and extract information from images at the pixel level. 
- Image Segmentation can be used for object recognition and localization which offers tremendous value in many applications such as medical imaging and self driving cars. 
- The goal is to train a Neural Network to produce pixel-wise mask of the image.
- Modern image segmentation techniques are based on deep learning approach which makes use of common architectures such as CNN, FCN.
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
- The first CNN layers are used to extract high level general features.
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
- We want the model to generalize the data and not to memorize the data.
- If error on training data and validation data is going down then the model is able to generalize the data.
- 3 generators: train generator, test generator, and validation generator.
- Feed images in batches of 16 to both.
- We shuffle the images so that the model does not memorize the ordering of images.
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
- Download the model that has been pretrained on imagenet data set.
- We don't include the dense layer on the end as we will add our own.
- We can overcome the vanishing gradient problem by having multiple layers stacked on top of each other.
- We freeze the model weigths.
- Add an average pooling 2D layer, flatten it, add a dense layer with 256 neurons, and dropout for 30% which removes co-dependecy between layers.
![Screenshot (116)](https://user-images.githubusercontent.com/70371572/131225544-60a14ba5-168e-4c28-9e91-aecb823381a3.png)
![Screenshot (117)](https://user-images.githubusercontent.com/70371572/131225547-6a107f81-4091-4207-8e65-6f716aee3012.png)

- We compile the model using Adam Optimizer.
``` model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics= ["accuracy"])```
-  We use early stopping to exit training if validation loss is not decreasing even after certain epochs (patience)
```earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)```

- Then we save the best model with least validation loss
```checkpointer = ModelCheckpoint(filepath="classifier-resnet-wei```

- We also save the model architecture to json file for future use
```
model_json = model.to_json()
with open("classifier-resnet-model.json","w") as json_file:
  json_file.write(model_json)
```


### ASSESS TRAINED MODEL PERFORMANCE
- We open the already saved the architecture. We load the presaved model.
- Now we feed in the data the model has never seen before 
```test_predict = model.predict(test_generator, steps = test_generator.n // 16, verbose =1)```
- As we have a binary classifier model so we can at this stage make out if there is a tumor or not.
![Screenshot (118)](https://user-images.githubusercontent.com/70371572/131228996-ba211e40-5e73-433e-9774-c8559d77b8bb.png)
![Screenshot (119)](https://user-images.githubusercontent.com/70371572/131229000-aab00f22-35ec-424e-9273-106180966aec.png)
- We are going to compare predicted and actual values of tumor or not.
- Generate Accuracy score: ```0.9809027777777778```, and Confusion matrix.
![Screenshot (120)](https://user-images.githubusercontent.com/70371572/131228986-a977dba5-7e21-4dfa-9eab-797b528e4549.png)

## Part 6. THeory behind the ResUNet Models

RESUNET
- ResUNet architecture combines UNet backbone architecture with residual blocks to overcome the vanishing gradients problem present in deep architectures.
- Unet architecture is based on Fully Convolutional Networks and modified in a way that it performs well on segmentation tasks.
- Resunet consists of 3 parts:
  - Encoder
  - Bottleneck
  - Decoder

![Screenshot (121)](https://user-images.githubusercontent.com/70371572/131229498-9b517a01-f020-4cce-a921-2b3cb0527ce4.png)
![Screenshot (122)](https://user-images.githubusercontent.com/70371572/131229507-6f4f6ea7-7a9e-42ec-9755-168027d4e3b7.png)
![Screenshot (123)](https://user-images.githubusercontent.com/70371572/131229513-b18e9745-ec2e-4d67-ab59-af3a50a468db.png)

## Part 7.  Build a segmentation Model to localize Tumor
- We only use the scans with mask column = 1 i.e. which have tumor.
- Then we split the data into train and test data.
- X_val is divided into validation and testing data.
- We create separate lists for Image ID and classID and use it to train RESUNETS. The input would be image and output would be mask associated with the image.
- We use the generator from utlities. Now we build the ResUNets. This is done by first defining Resblock.
- 2 paths are there in this block: main and short path with a relu activation function. 
  - main path: Read more about he_normal: https://medium.com/@prateekvishnu/xavier-and-he-normal-he-et-al-initialization-8e3d7a087528
  - Short path: Read more here: https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33
### Train the model to localize tumor
- Loss function:
We need a custom loss function to train this ResUNet.So, we have used the loss function as it is from https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py
- File contains the code for pixel level segmentation. We use Adam optimizer with earlystopping, and the save the best model, set epochs to 1. 

###  Assess TRAINED SEGMENTATION RESUNET MODEL Performance
- Load the actual architecture, and weights. We have a costumn loss function, data generator and prediction.
- Models will make prediction for testing data and returns if tumor or not. If there is a tumor then the predicted mask is also shown.
- 5 columns: MRI scan, Original Mask, Mask generated by RESUNET, overlap of MRI scan with the original mask and overlap of MRI scan with the predicted mask.
![Screenshot (126)](https://user-images.githubusercontent.com/70371572/131230338-0f08a739-d912-4dfd-a520-d17c40f87264.png)
![Screenshot (125)](https://user-images.githubusercontent.com/70371572/131230302-f295112d-d4bc-4477-b9be-5c29be8fb100.png)

As we can see the predicted masks for tumor is very close to the actual tumor location. Therefore this model was successful in detecting and predicting tumors in brain.
Note: This project was with help of online Udemy course.
