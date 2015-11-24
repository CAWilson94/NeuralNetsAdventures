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
# 33% data for testing remaining 67% for training
(trainX, testX, trainY, testY) = train_test_split(
    dataset.data / 255.0, dataset.target.astype("int0"), test_size=0.33)

"""
Train the network with 784 inputs (flattened 28x28 greyscale image)
300 hidden units, 10 output units: 1 for each possible output
classification,which are the digits 1-10)
"""
dbn = DBN(
    [trainX.shape[1], 300, 10],
    learn_rates= 0.3,
    learn_rate_decays= 0.9,
    epochs =10,
    verbose =1)
dbn.fit(trainX, trainY)
