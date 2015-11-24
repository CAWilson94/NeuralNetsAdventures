"""
Created November 2015 
@author Charlotte Alexandra Wilson

"""

# Import packages needed
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from nolearn.dbn import DBN
import numpy as np
import cv2

# Use the MNIST dataset
print "[X]downloading dataset..."
dataset = datasets.fetch_mldata("MNIST Original")

"""
Generate training and testing splits!
"""

# scale the data to range [0,1] 
# construct training and testing split
(trainX, testX, trainY, testY) = train_test_split(
    dataset.data / 255.0, dataset.target.astype("int0"), test_size = 0.33)
