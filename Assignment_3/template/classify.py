# classify.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set

return - a list containing predicted labels for dev_set
"""

import numpy as np
from sklearn.linear_model import Perceptron

def predict(W,b,X):
    return 1 if ((np.dot(W, X)+b) > 0) else 0

def accuracy(W,b,X,y):
    y_pred = []
    for i in range(0,len(X)):
        y_pred.append(predict(W,b,X[i]))
    return np.sum(np.equal(y,y_pred))/len(X)

def trainPerceptron(train_set, train_labels, lr, max_iter):
    # TODO: Write your code here
    max_accuaracy = 0
    for i in range(0,max_iter):
        W = np.zeros(train_set.shape[1])
        b=0
        for X,y in zip(train_set,train_labels):
            y_pred = predict(W,b,X)
            if y == 1 and y_pred == 0:
                W = W + lr * X
                b = b - lr * 1
            elif y == 0 and y_pred == 1:
                W = W - lr * X
                b= b + lr * 1
        acc = accuracy(W,b,train_set,train_labels)
        if  acc > max_accuaracy:
            max_accuaracy = acc
            W_max = W
            b_max = b
        

    # return the trained weight and bias parameters
    return W_max, b_max

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    W , b = trainPerceptron(train_set,train_labels,learning_rate,max_iter)
    Y_pred = []
    for i in range(0,len(dev_set)):
        Y_pred.append(predict(W,b,dev_set[i]))
    # Train perceptron model and return predicted labels of development set

    return Y_pred


