#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : lr
Author          : Charles Young
Python Version  : Python 3.6.1
Date            : 2018-10-17
'''

from scipy.stats import mode
import numpy as np

class KNN:

	def __init__(self, n_neighbors):
		self.n_neighbors = n_neighbors

	def fit(self, features, labels):
		# Simply store the training data
		self.features = features
		self.labels = labels

	def predict_one_sample(self, feature):
		diff = (self.features - feature)
		# Calculate row-wise inner product
		dst = np.einsum('ij, ij->i', diff, diff)
		nearest = self.labels[np.argsort(dst)[:self.n_neighbors]]
		return mode(nearest)[0][0]

	def predict(self, features):
		return np.apply_along_axis(self.predict_one_sample, 1, features)