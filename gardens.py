################################################################################
# Author:   Evan Dietrich
# Course:   Comp 131 - Intro AI
# Prof:     Santini
#
# Assign:   Artificial Neural Networks
# Date:     12/16/2020
# File:     gardens.py
################################################################################

################################################################################
#       IMPORTS + GLOBALS
################################################################################

FILE = 'data.txt'
LEARN_RATE = 0.12
NUM_EPOCHS = 300

import os
import sys
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

################################################################################
#       MODEL FUNCTIONALITY + HELPERS
################################################################################

# File I/O, isolating data into relevant characteristics and claass
def PreProcessing():
    iris = pd.read_csv(FILE, sep=",", header=None)
    iris.columns = ['SepLenCm', 'SepWthCm', 'PetLenCm', 'PetWidCm', 'Class']
    x_set = np.array(iris[['SepLenCm', 'SepWthCm', 'PetLenCm', 'PetWidCm']])
    y_set = OneHotEncoder(sparse=False).fit_transform(np.array(iris.Class).reshape(-1, 1))
    return x_set, y_set

# Randomizes init weightings, ensures expectation of stochastic gradient descent
def InitWeighting(vertices):
    layers, weightings = len(vertices), []
    for x in range(1, layers):
        wght = [[np.random.uniform(-1, 1) for y in range(vertices[x-1]+1)]
              for z in range(vertices[x])]
        weightings.append(np.matrix(wght))
    return weightings

# Trains network every epoch, updating weightings accordingly
def NeuralNet(xtrain, ytrain, xvalid, yvalid, vertices):
    weightings = InitWeighting(vertices)
    for epoch in range(1, NUM_EPOCHS+1):
        weightings = TrainNet(xtrain, ytrain, weightings)
        if(epoch % 30 == 0):
            print("\nEpoch: "+str(epoch))
            print("Train Accuracy: "+str(Accuracy(xtrain, ytrain, weightings)))
            print("Valid Accuracy: "+str(Accuracy(xvalid, yvalid, weightings)))
    return weightings

# Sigmoid Activation Function
# Alternatively, could use another traditional activation such as ReLU function
def SigMap(val):
    return (1/(1+np.exp(-val)))

# Network activates on an instance, calc's error of each neuron in output layer
def ForwardProp(x, weightings, layers):
    activations, vector = [x], x
    for i in range(layers):
        activation = SigMap(np.dot(vector, weightings[i].T))
        activations.append(activation)
        vector = np.append(1, activation)
    return activations

# Propagates error backward, removing bias of prev layers' weightings
def BackwardProp(y, activations, weightings, layers):
    curr_err = np.matrix(y-activations[-1])
    for i in range(layers, 0, -1):
        curr_activ = activations[i]
        if(i > 1):
            prev = np.append(1, activations[i-1])
        else:
            prev = activations[0]
        delt = np.multiply(curr_err, (np.multiply(curr_activ, 1-curr_activ)))
        weightings[i-1] += (LEARN_RATE * np.multiply(delt.T, prev))
        curr_err = np.dot(delt, (np.delete(weightings[i-1], [0], axis = 1)))
    return weightings

# Updates weightings on iris-data post-forward/backward pass
def TrainNet(x_set, y_set, weightings):
    layers = len(weightings)
    for i in range(len(x_set)):
        x, y = x_set[i], y_set[i]
        x = np.matrix(np.append(1, x))
        activations = ForwardProp(x, weightings, layers)
        weightings = BackwardProp(y, activations, weightings, layers)
    return weightings

# Predicts a flower-type from dataset's classes based on statistical confidence
# value determined after passing object through the network, max's activation
def Predict(obj, weightings):
    obj, layers = np.append(1, obj), len(weightings)
    activations = ForwardProp(obj, weightings, layers)
    resultant = activations[-1].A1
    max_activ, indx = resultant[0], 0
    for i in range(1, len(resultant)):
        if(resultant[i] > max_activ):
            max_activ, indx = resultant[i], i
    y, y[indx] = [0 for i in range(len(resultant))], 1
    return y

# Calculates accuracy based on % correct predictions
def Accuracy(x_set, y_set, weightings):
    correct = 0
    for i in range(len(x_set)):
        x, y = x_set[i], list(y_set[i])
        if (y == (Predict(x, weightings))):
            correct += 1
    return (correct/len(x_set))

################################################################################
#       MAIN PROGRAM
################################################################################

if __name__ == '__main__':
    print("\n>>> Running ---Neural Network--- on: '"+FILE+"'"+"\n")

    x_set, y_set = PreProcessing()
    ts1, ts2 = .15, .1
    xtrain, xtest, ytrain, ytest = train_test_split(x_set, y_set, test_size=ts1)
    xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, test_size=ts2)
    n_features, n_classes = len(x_set[0]), len(y_set[0])
    layers = [n_features, 5, 10, n_classes]
    weightings = NeuralNet(xtrain, ytrain, xvalid, yvalid, layers);

    print("\nTesting Accuracy: " + str(Accuracy(xtest, ytest, weightings)))
    print(">>> Runtime complete <<<\n")
