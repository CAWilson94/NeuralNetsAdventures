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
# Input layer: input node for each entry in vector list
# Input layer feeds forward into hidden layer.
# Hidden layer represented by RBM
# Output of 300 nodes from hidden layer fed into output layer
# Output layer consist of an output for each of the class labels.
dbn = DBN(
    # [[numNodes input layer], numNodes hidden layer, numNodes output layer ]
    [trainX.shape[1], 300, 10],
    # Learning rate of algorithm
    learn_rates=0.3,
    # Decay of learn rate
    learn_rate_decays=0.9,
    # Iterations of training data (epochs)
    epochs=10,
    # Verbosity level
    verbose=1)
dbn.fit(trainX, trainY)  # TRAINING IS HERE!!

# Both learn_rates and learn_rates_decays can be specified as a single
# floating point values or list of said values.
# If you specify a single value this learning rate/decay rate will be 
# applied to ALL layers in the network.
# If you specify a list of values, the corresponding learning and decay
# rate will be used for those respective layers

"""
The network is trained, time to evaluate it!

"""

# Computer predictions for test data 
# Show a classification report

preds = dbn.predict(testX)
print classification_report(testY, preds)  # Table of accuracies 

"""
Inspect individually rather than on aggregate 
as a further demonstration of the network

"""

# Randomly select a few of the test instance
for i in np.random.choice(np.arange(0, len(testY)), size=(10,)) :
    # classify the digit
    pred = dbn.predict(np.atleast_2d(testX[i]))
    
    # reshape feature vector to be a 28x28 pixel image,
    # change data type to be an unsigned 8-bit integer
    image =(testX[i] * 255).reshape((28,28)).astype("uint8")
    
    #Show image and prediction
    print "Actual digit is {0}, predicted {1}".format(testY[i],pred[0])
    cv2.imshow("Digit",image)
    cv2.waitKey(0)
    
         
