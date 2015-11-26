'''
Created on 26 Nov 2015
@author: Charlotte Alexandra Wilson
 
Each given file represents one seconds worth of data
Rows: high voltage cycle number (0 - 49) ; power frequency = 50HZ, therefore 1s of data
Columns: different wave position on HV waveform 
There are 64 columns thus, 360 degrees/64 = 5.625 degrees per column

So, if a PD occurred during the second cycle at 10 degrees, it would go in the second row and second column.
Amplitude of the PD pulse is represented by the colour intensity of the pixel(in bmp) or the value of the cell(csv)

Input to neural nets: One of the csv files
There are 64x50 values in the file thus, number of inputs to network = 64x50 = 3200
The number of outputs will be 6 for the 6 different defect types.

'''
import pandas as pd 
import numpy as np
# Neural nets model inputs
from nolearn.dbn import DBN 
import cv2
# Scikit learn imports
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

print "hello"

# Import the csv file
#---------------------
train_data = pd.read_csv(open('BC27.L3D.csv'))

# Generate training data
#-----------------------

# Convert into pandas data frame
dataset = pd.DataFrame(train_data)
print "panda data frame"
# Get data into a bunch object
type(dataset)
print "bunches"

# Training the network
#---------------------

"""
Train the network with 3200 inputs (64x50 values in file)
6 output units (for the different defects)
Lets start with 2 hidden units
"""

# Numbers from example used here
(trainX, testX, trainY, testY) = train_test_split(
    dataset / 255.0, dataset.target.astype("int0"), test_size=0.33)

dbn = DBN(
    # [[numNodes input layer], numNodes hidden layer, numNodes output layer ]
    [trainX.shape[1], 2, 6],
    # Learning rate of algorithm
    learn_rates=0.3,
    # Decay of learn rate
    learn_rate_decays=0.9,
    # Iterations of training data (epochs)
    epochs=10,
    # Verbosity level
    verbose=1)
dbn.fit(trainX,trainY) 

print "trained yo!"

# Evaluate network
#-----------------
preds = dbn.predict(testX)
print classification_report(testY, preds) # Table of accuracies 







