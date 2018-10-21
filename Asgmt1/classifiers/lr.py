#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : lr
Author          : Charles Young
Python Version  : Python 3.6.1
Requirements    : (Please check document: requirements.txt or use command "pip install -r requirements.txt")
Date            : 2018-10-17
'''

import numpy as np
from sklearn.preprocessing import LabelEncoder

class lr:

	def __init__(self, num_steps = 30, learning_rate = 5e-5):
		self.num_steps = num_steps
		self.learning_rate = learning_rate
		self.lben = LabelEncoder()
		self.num_classes = None
		self.weights = None
		self.multclass = False

	def sigmoid(self, scores):
		return 1 / (1 + np.exp(-scores))

	def log_likelihood(self, features, target):
		scores = np.dot(features, self.weights)
		return np.sum(target * scores - np.log(1 + np.exp(scores)))

	def fit(self, features, target):
		features = np.hstack((np.ones((features.shape[0], 1)), features))

		label_en = self.lben.fit_transform(target.flatten())

		labelSet = np.unique(label_en)
		self.num_classes = len(labelSet)

		if self.num_classes > 2:
			self.multclass = True

		if self.multclass: # One vs Rest
			self.weights = np.zeros((self.num_classes, features.shape[1]))

			def labelBinarize(labels, labelSet):
				newLabels = np.tile(labels, [len(labelSet), 1])
				for idx in range(len(labelSet)):
					newLabels[idx,][newLabels[idx,] != labelSet[idx]] = -1
					newLabels[idx,][newLabels[idx,] == labelSet[idx]] = +1
				newLabels[newLabels == -1] = 0
				return newLabels

			# Generate labels for each binary classifier
			label_matrix = labelBinarize(label_en, labelSet)

			for step in range(self.num_steps):
				# Fitting each classifier
				for n_class in range(self.num_classes):
					predictions = self.sigmoid(np.dot(features, self.weights[n_class,]))

					# Update weights: optimize through gradient descent algorilthm
					err = label_matrix[n_class,] - predictions
					gradient = np.dot(features.T, err)
					self.weights[n_class,] += self.learning_rate * gradient

		else:
			self.weights = np.zeros(features.shape[1])

			for step in range(self.num_steps):
				predictions = self.sigmoid(np.dot(features, self.weights))

				# Update weights: optimize through gradient descent algorilthm
				err = target - predictions
				gradient = np.dot(features.T, err)
				self.weights += self.learning_rate * gradient

	def predict_prob(self, features):
		features = np.hstack((np.ones((features.shape[0], 1)), features))

		if self.weights is None:
			raise "Call prediction before fitting."

		if self.weights.ndim == 1 and features.shape[1] != self.weights.size or self.weights.ndim == 2 and features.shape[1] != self.weights.shape[1]:
			raise "The dimension of the input data is not matched to the pre-trained model."

		# If multiclass, vote
		if self.weights.ndim == 2:
			predictions = np.zeros((self.num_classes, features.shape[0]))
			for n_class in range(self.num_classes):
				predictions[n_class,] = self.sigmoid(np.dot(features, self.weights[n_class,]))
			predictions /= predictions.sum(axis = 1)[:, None]
			return predictions.T
		# else if binary class
		else:
			return self.sigmoid(np.dot(features, self.weights))

	def predict(self, features):
		predictions = self.predict_prob(features)
		
		# If multiclass, choose the label with the largest prob
		if self.weights.ndim == 2:
			return self.lben.inverse_transform(np.argmax(predictions, axis = 1))
		else:
			return self.lben.inverse_transform((np.round(predictions).astype('int64')))
