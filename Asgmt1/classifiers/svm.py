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
        self.tmp_alpha = None
        self.tmp_bias = None
        self.E_cache = None

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
            tmp_bias = b1
        elif 0 < alpha_j < self.C:
            tmp_bias = b2
        else:
            tmp_bias = (b1 + b2)/2
        self.E_cache -= self.tmp_bias - tmp_bias # Update error cache
        return tmp_bias

    # given \alpha_i, generate the best \alpha_j
    def generate_j(self, i, Ei):
        self.E_cache[i] = Ei

        #  choose the alpha that gives the maximum delta E
        delta_Es = np.abs(self.E_cache - Ei)
        j = np.argmax(delta_Es)
        Ej = self.E_cache[j]

        return j, Ej

    """
        Core functions:
    """
    # SMO inner loop
    def smo_inner(self, X, y, i, kernel):
        """
        The inner loop of SMO algorithm
        :return: whether changed alpha values. 0 = unchanged, 1 = changed (go to function smo to see why)
        """

        # Sequential Minimal Optimization (SMO):
        # 1. Initialize all \alpha to zero
        # 2. Randomly select a pair of \alpha_i and \alpha_j
        # 3. Calculate the upper and the lower limits of \alpha
        # 4. Update \alpha_i and \alpha_j:
        # 5. Update bias
        # 6. Repeat 2-5 until stop

        # Compute prediction errors
        E_i = self.calc_e(X[i], y[i], self.tmp_alpha, X, y, self.tmp_bias, kernel) 

        if ((y[i] * E_i < -1e-5) and (self.tmp_alpha[i] < self.C)) or \
            ((y[i] * E_i > 1e-5) and (self.tmp_alpha[i] > 0)):
            j, E_j = self.generate_j(i, E_i)

            # Record some variables
            X_i, X_j, y_i, y_j = X[i,:].copy(), X[j,:].copy(), y[i].copy(), y[j].copy()
            alpha_old_j, alpha_old_i = self.tmp_alpha[j].copy(), self.tmp_alpha[i].copy()

            # Compute the upper and lower limits of \alpha
            (L, H) = self.calc_LH(self.C, alpha_old_j, alpha_old_i, y_j, y_i)
            if L == H:
                return 0

            # Calculate the value of \eta
            eta = kernel(X_i, X_i) + kernel(X_j, X_j) - 2 * kernel(X_i, X_j)
            if eta <= 0:
                return 0

            # Update \alpha values
            self.tmp_alpha[j] = alpha_old_j + float(y_j * (E_i - E_j)) / eta
            self.tmp_alpha[j] = min(self.tmp_alpha[j], H)
            self.tmp_alpha[j] = max(self.tmp_alpha[j], L)
            # Update prediction error
            self.E_cache[j] = self.calc_e(X_j, y_j, self.tmp_alpha, X, y, self.tmp_bias, kernel)   
            if abs(self.tmp_alpha[j] - alpha_old_j) < 0.00001:
                # j not moving enough
                return 0
            self.tmp_alpha[i] = alpha_old_i + y_i * y_j * (alpha_old_j - self.tmp_alpha[j])
            # Update prediction error
            self.E_cache[i] = self.calc_e(X_i, y_i, self.tmp_alpha, X, y, self.tmp_bias, kernel) 

            # Compute bias value
            self.tmp_bias = self.calc_b(self.tmp_bias, X_i, X_j, y_i, y_j, E_i, E_j, self.tmp_alpha[i], self.tmp_alpha[j], alpha_old_i, alpha_old_j, kernel)

            return 1

        else:
            return 0

    # fit a binary classifier using SMO
    def fit_binary(self, X, y):

        n_sample, n_feature = X.shape[0], X.shape[1]
        kernel = self.kernels[self.kernel]

        iter = 0
        self.tmp_alpha = np.zeros((n_sample))
        self.tmp_bias = 0

        entire_set = True
        alpha_pairs_changed = 0 

        while (iter < self.max_iter) and ((alpha_pairs_changed > 0) or entire_set):
            alpha_pairs_changed = 0
            if entire_set:  # go over all
                for i in range(n_sample):
                    alpha_pairs_changed += self.smo_inner(X, y, i, kernel)
                iter += 1
            else:  # go over non-bound (railed) alphas
                non_bound_i = np.nonzero((self.tmp_alpha > 0) * (self.tmp_alpha < self.C))[0]
                for i in non_bound_i:
                    alpha_pairs_changed += self.smo_inner(X, y, i, kernel)
                iter += 1
            if entire_set:
                entire_set = False  # toggle entire set loop
            elif alpha_pairs_changed == 0:
                entire_set = True
        return iter, self.tmp_alpha, self.tmp_bias

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
                self.E_cache = (-label_matrix[n_class,]).astype('float64')
                it, alpha, bias = self.fit_binary(features, label_matrix[n_class,])
                self.alpha.append(alpha)
                self.bias.append(bias)

            self.alpha = np.array(self.alpha)
            self.bias = np.array(self.bias)

        else:
            self.y = label_en
            self.E_cache = (-label_en).astype('float64')
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