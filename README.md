# BrainTumour_Detection
A project that detects brain Tumors using layered Deep Learning Pipeline to perform Classification and Segmentation. Deep learning has been proven to be superior in detecting diseases from X-rays, MRI scans which could significantly improve the speed and accuracy of diagnosis. We have 3929 Brain scans along with their tumour location. 

# Approach
- Goal of image segmentation is to understand and extract information from images at the pixel level. 
- Image Segmentation can be used for object recognition and localization which offers tremendous value in many applications such as medical imaging and self driving cars. 
- The goal is to train a Neural Network to produce pixel-wise mask of the image.
- Mordern image segmentation techniques are based on deep learning approach which makes use of common architectures such as CNN, FCN.
- We will use ResUNet architecture to solve the task.

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

