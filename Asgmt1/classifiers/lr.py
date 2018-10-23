#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : lr
Author          : Charles Young
Python Version  : Python 3.6.1
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

	def sigmoid(self, scores):
		return 1 / (1 + np.exp(-scores))

	# def log_likelihood(self, features, targets):
	# 	scores = np.dot(features, self.weights)
	# 	return np.sum(targets * scores - np.log(1 + np.exp(scores)))

	def fit(self, features, targets):
	# Parameters:
	#	features -- np.ndarray: 
	#		input features matrix shaped as (samples, features)
	# 	targets -- np.ndarray:
	#		input labels vector shaped as (samples,) 
	# Return:
	# 	None

		# Add a column of 1 to combine parameter bias with parameter weights
		features = np.hstack((np.ones((features.shape[0], 1)), features))

		# Fit a label encoder
		label_en = self.lben.fit_transform(targets.flatten())

		# Get unique label set and calc the number of classes
		labelSet = np.unique(label_en)
		self.num_classes = len(labelSet)

		# If multiclass, apply one vs rest method
		if self.num_classes > 2:
			# Initialize weights
			self.weights = np.zeros((self.num_classes, features.shape[1]))

			# helper func to generate labels for each binary classifier
			def labelBinarize(labels, labelSet):
				newLabels = np.tile(labels, [len(labelSet), 1])
				for idx in range(len(labelSet)):
					newLabels[idx,][newLabels[idx,] != labelSet[idx]] = -1
					newLabels[idx,][newLabels[idx,] == labelSet[idx]] = +1
				newLabels[newLabels == -1] = 0
				return newLabels

			# Generate labels for each binary classifier
			label_matrix = labelBinarize(label_en, labelSet)

			# Iteration
			for step in range(self.num_steps):
				# Specify num_classes classifiers and fit each classifier
				for n_class in range(self.num_classes):
					predictions = self.sigmoid(np.dot(features, self.weights[n_class,]))

					# Update weights: optimize through gradient descent algorilthm
					err = label_matrix[n_class,] - predictions
					gradient = np.dot(features.T, err)
					self.weights[n_class,] += self.learning_rate * gradient
		# If binary-class
		else:
			# Initialize weights
			self.weights = np.zeros(features.shape[1])

			for step in range(self.num_steps):
				predictions = self.sigmoid(np.dot(features, self.weights))

				# Update weights: optimize through gradient descent algorilthm
				err = targets - predictions
				gradient = np.dot(features.T, err)
				self.weights += self.learning_rate * gradient

	def predict_prob(self, features):
		# Parameters:
		#	features -- np.ndarray: 
		#		input features matrix shaped as (samples, features)
		# Return:
		# 	predictions -- np.ndarray:
		#		if multiclass, a matrix shaped as (samples, num_classes), (i,j) denotes the probability that sample_i belong to class_j
		#		elif binary class, a vector shaped as (samples, ), (i) denotes the probability that sample_i belong to positive class

		features = np.hstack((np.ones((features.shape[0], 1)), features))

		if self.weights is None:
			raise "Call prediction before fitting."

		if self.weights.ndim == 1 and features.shape[1] != self.weights.size or self.weights.ndim == 2 and features.shape[1] != self.weights.shape[1]:
			raise "The dimension of the input data is not matched to the pre-trained model."

		# If multiclass, apply all classifiers to a sample and calc the mean of all the predictions
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
		# Parameters:
		#	features -- np.ndarray: 
		#		input features matrix shaped as (samples, features)
		# Return:
		# 	predictions -- np.ndarray:

		predictions = self.predict_prob(features)
		
		# If multiclass, choose the label with the largest prob and decode labels
		if self.weights.ndim == 2:
			return self.lben.inverse_transform(np.argmax(predictions, axis = 1))
		else:
			return self.lben.inverse_transform((np.round(predictions).astype('int64')))
