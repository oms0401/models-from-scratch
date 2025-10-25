"""
The Core Idea 🧠The goal of linear regression is to find the "best-fit line" (or plane, for multiple features) that describes the relationship between your input features (X) and your output target (y).You're trying to find the parameters for this simple equation y = mx + b In your code, which is built to handle multiple features.
"""

import numpy as np


class LinearRegression():

    def __init__(self,lr=0.001,n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        self.weights=None
        self.bias=None
    
    def fit(self,X,y):
        n_samples,n_features=X.shape #(100,3)

        self.weights=np.zeros(n_features) #3
        self.bias=0

        for _ in range(self.n_iters):

            y_pred=np.dot(X,self.weights)+self.bias

            dw = (1/n_samples)*np.dot(X.T,(y_pred-y))
            db = (1/n_samples)*np.sum((y_pred-y))

            
            self.weights-=self.lr*dw
            self.bias-=self.lr*db


    def predict(self,X):
       y_approx=np.dot(X,self.weights)+self.bias
       return y_approx