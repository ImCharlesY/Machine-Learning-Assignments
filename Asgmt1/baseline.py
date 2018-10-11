#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : baseline
Author          : Charles Young
Python Version  : Python 3.6.1
Requirements    : (Please check document: requirements.txt or use command "pip install -r requirements.txt")
Date            : 2018-10-06
'''
print(__doc__)

# import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
import numpy as np
import os
import sys
import argparse
from sklearn import svm, metrics
from mnist_helper import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--train_size', type = int, default = 600, help = "Size of train dataset.")
parser.add_argument('--test_size', type = int, default = 100, help = "Size of test dataset.")
parser.add_argument('-C', type = int, default = 5, help = "Parameter C for svm classifier.")
parser.add_argument('--kernel', default = 'rbf', help = "Kernel used in svm classifier.")
parser.add_argument('--gamma', type = float, default = .05, help = "Parameter gamma for svm classifier.")
args = parser.parse_args()

# The digits dataset
train, test = get_dataset('./data/')

# Split features and labels
train_x = train[:,1:-1]
train_y = train[:,-1]
del train
test_x = test[:,1:-1]
test_y = test[:,-1]
del test

print('Shape of train dataset: {}'.format(train_x.shape))
print('Shape of test dataset: {}'.format(test_x.shape))

# Create a classifier: a support vector classifier
print('Create classifier ...')
classifier = svm.SVC(C = args.C, kernel = args.kernel, gamma = args.gamma)

# We learn the digits on the first half of the digits
print('Start fitting ...')
classifier.fit(train_x[:args.train_size,] / 255.0, train_y[:args.train_size,])

# Now predict the value of the digit on the second half:
expected = test_y[:args.test_size,]
predicted = classifier.predict(test_x[:args.test_size,] / 255.0)

print('Classification report for classifier %s:\n%s\n'
      % (classifier, metrics.classification_report(expected, predicted)))
print('Confusion matrix:\n%s' % metrics.confusion_matrix(expected, predicted))
print('Accuracy: {}'.format(metrics.accuracy_score(expected, predicted)))
