"""Logistic regression model."""

import numpy as np
from sklearn.metrics import accuracy_score

class Logistic:
    def __init__(self, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = 0.5

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        lst = np.empty(len(z))
        i=0
        for x in z:
            if x >= 0:
                z1 = np.exp(-x)
                lst[i]= 1 / (1 + z1)
            else:
                z1 = np.exp(x)
                lst[i] = z1 / (1 + z1)
            i=i+1
        return lst


    def gradient(self, x, y_true, y_pred):
        diff =  y_pred - y_true
        grad_b = np.mean(diff)
        grad_w = np.matmul(x.transpose(), diff)
        grad_w = np.array([np.mean(grad) for grad in grad_w])
        return grad_w, grad_b

        

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        self.w = np.zeros(X_train.shape[1])
        self.b = 0
        for i in range(self.epochs):
            x_w = np.matmul(self.w, X_train.transpose()) + self.b
            y = self.sigmoid(x_w)
            grad_w, grad_b = self.gradient(X_train, y_train, y)
            self.w = self.w - self.lr * grad_w
            self.b = self.b - self.lr * grad_b


    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        x_w = np.matmul(X_test, self.w.transpose()) + self.b
        prob = self.sigmoid(x_w)
        return [1 if p > self.threshold else 0 for p in prob]
        
