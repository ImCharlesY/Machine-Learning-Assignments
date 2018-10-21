#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : svm 
Author          : Charles Young
Python Version  : Python 3.6.1
Requirements    : (Please check document: requirements.txt or use command "pip install -r requirements.txt")
Date            : 2018-10-17
'''

import numpy as np
from sklearn.preprocessing import LabelEncoder

class svm:
    """
        Simple implementation of a Support Vector Machine using the
        Sequential Minimal Optimization (SMO) algorithm.
    """
    def __init__(self, max_iter = 10000, kernel = 'rbf', C = 1.0, gamma = 0.05, epsilon = 0.001, random_state = 0):
        self.kernels = {
            'linear' : self.kernel_linear,
            'quadratic' : self.kernel_quadratic,
            'rbf' : self.kernel_gaussian
        }

        self.max_iter = max_iter
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.epsilon = epsilon

        self.lben = LabelEncoder()

        self.X = None
        self.y = None

        self.num_classes = None

        self.alpha = None
        self.bias = None

        np.random.seed(random_state)

    """
    Define kernel functions:
    """
    def kernel_linear(self, x, xk):
        return np.dot(x, xk)
    def kernel_quadratic(self, x, xk):
        return np.dot(x, xk) ** 2
    def kernel_gaussian(self, x, xk):
        if len(x) == len(xk):
            return np.exp(- self.gamma * (np.linalg.norm(x - xk) ** 2))
        else:
            return np.exp(- self.gamma * (np.linalg.norm(x - xk, axis = 1) ** 2))

    """
    Define some help functions:
    """
    # Calculate limits of \alpha
    def calc_LH(self, C, alpha_old_j, alpha_old_i, y_j, y_i):
        if(y_i != y_j):
            return (max(0, alpha_old_j - alpha_old_i), min(C, C - alpha_old_i + alpha_old_j))
        else:
            return (max(0, alpha_old_i + alpha_old_j - C), min(C, alpha_old_i + alpha_old_j))   

    # Calculate prediction
    def calc_pred(self, test, X, y, alpha, b, kernel):
        return (alpha * y * kernel(X, test)).sum() + b 

    # Calculate prediction error
    def calc_e(self, x_k, y_k, alpha, X, y, b, kernel):
        return self.calc_pred(x_k, X, y, alpha, b, kernel) - y_k

    # Calculate bias
    def calc_b(self, b, X_i, X_j, y_i, y_j, E_i, E_j, alpha_i, alpha_j, alpha_old_i, alpha_old_j, kernel):
        b1 = b - E_i - y_i * (alpha_i - alpha_old_i) * (kernel(X_i, X_i)) - y_j * (alpha_j - alpha_old_j) * (kernel(X_i, X_j))
        b2 = b - E_j - y_i * (alpha_i - alpha_old_i) * (kernel(X_i, X_j)) - y_j * (alpha_j - alpha_old_j) * (kernel(X_j, X_j))

        if 0 < alpha_i < self.C:
            return b1
        elif 0 < alpha_j < self.C:
            return b2
        else:
            return (b1 + b2)/2

    """
    Core functions:
    """
    # fit a binary classifier using SMO
    def fit_binary(self, X, y):

        # Sequential Minimal Optimization (SMO):
        # 1. Initialize all \alpha to zero
        # 2. Randomly select a pair of \alpha_i and \alpha_j
        # 3. Calculate the upper and the lower limits of \alpha
        # 4. Update \alpha_i and \alpha_j:
        # 5. Update bias
        # 6. Repeat 2-5 until stop

        n_sample, n_feature = X.shape[0], X.shape[1]
        kernel = self.kernels[self.kernel]

        # Initialization
        alpha = np.zeros((n_sample))
        bias = 0
        it = 0

        while True:
            it += 1
            alpha_old = np.copy(alpha)

            for j in range(0, n_sample):
                # Randomly select i that is not equal to j
                i = j
                while (j == i):
                    i = np.random.randint(0, n_sample) 

                # Record some variables
                X_i, X_j, y_i, y_j = X[i,:], X[j,:], y[i], y[j]
                alpha_old_j, alpha_old_i = alpha[j], alpha[i]

                # Compute the upper and lower limits of \alpha
                (L, H) = self.calc_LH(self.C, alpha_old_j, alpha_old_i, y_j, y_i)
                if L == H:
                    continue

                # Calculate the value of \eta
                eta = kernel(X_i, X_i) + kernel(X_j, X_j) - 2 * kernel(X_i, X_j)
                if eta <= 0:
                    continue

                # Compute prediction errors
                E_i = self.calc_e(X_i, y_i, alpha, X, y, bias, kernel)
                E_j = self.calc_e(X_j, y_j, alpha, X, y, bias, kernel)

                # Update \alpha values
                alpha[j] = alpha_old_j + float(y_j * (E_i - E_j)) / eta
                alpha[j] = min(alpha[j], H)
                alpha[j] = max(alpha[j], L)
                alpha[i] = alpha_old_i + y_i * y_j * (alpha_old_j - alpha[j])

                # Compute bias value
                bias = self.calc_b(bias, X_i, X_j, y_i, y_j, E_i, E_j, alpha[i], alpha[j], alpha_old_i, alpha_old_j, kernel)

            # If \alpha varies slightly, stop iteration
            if np.linalg.norm(alpha - alpha_old) < self.epsilon:
                break

            if it >= self.max_iter:
                print("Iteration number exceeded the max of %n_feature iterations" % (self.max_iter))
                return it

        return it, alpha, bias

    def fit(self, features, target):
        label_en = self.lben.fit_transform(target.flatten())

        labelSet = np.unique(label_en)
        self.num_classes = len(labelSet)

        if self.num_classes > 2:
            self.multclass = True
        else:
            label_en[label_en == 0] = -1

        self.X = features

        if self.multclass: # One vs Rest

            def labelBinarize(labels, labelSet):
                newLabels = np.tile(labels, [len(labelSet), 1])
                for idx in range(len(labelSet)):
                    newLabels[idx,][newLabels[idx,] != labelSet[idx]] = -1
                    newLabels[idx,][newLabels[idx,] == labelSet[idx]] = +1
                return newLabels

            # Generate labels for each binary classifier
            label_matrix = labelBinarize(label_en, labelSet)

            self.y = label_matrix

            self.alpha = []
            self.bias = []

            # Fitting each classifier
            for n_class in range(self.num_classes):
                it, alpha, bias = self.fit_binary(features, label_matrix[n_class,])
                self.alpha.append(alpha)
                self.bias.append(bias)

            self.alpha = np.array(self.alpha)
            self.bias = np.array(self.bias)

        else:
            self.y = label_en
            it, self.alpha, self.bias = self.fit_binary(features, label_en)

    def predict_binary(self, test_features, train_features, train_labels, alpha, bias, kernel):
        prediction = np.zeros(len(test_features))
        for i in range(len(test_features)):
            prediction[i] = self.calc_pred(test_features[i,], train_features, train_labels, alpha, bias, kernel)
        return prediction

    def predict_prob(self, features):
        if self.alpha is None:
            raise "Call prediction before fitting."

        if features.shape[1] != self.X.shape[1]:
            raise "The dimension of the input data is not matched to the pre-trained model."

        # If multiclass, vote
        if self.alpha.ndim == 2:
            predictions = np.zeros((self.num_classes, features.shape[0]))
            for n_class in range(self.num_classes):
                predictions[n_class,] = self.predict_binary(features, self.X, self.y[n_class,], self.alpha[n_class, ], self.bias[n_class], self.kernels[self.kernel])
                tmpmin = np.abs(predictions[n_class,].min())
                predictions[n_class,] += tmpmin
            predictions /= predictions.sum(axis = 1)[:, None]
            # print(self.alpha)
            return predictions.T
        # else if binary class
        else:
            return self.predict_binary(features, self.X, self.y, self.alpha, self.bias, self.kernels[self.kernel])
        
    def predict(self, features):
        predictions = self.predict_prob(features)
        # If multiclass, choose the label with the largest prob
        if self.alpha.ndim == 2:
            return self.lben.inverse_transform(np.argmax(predictions, axis = 1))
        else:
            predictions[predictions > 0] = 1
            predictions[predictions <= 0] = 0
            return self.lben.inverse_transform(predictions)