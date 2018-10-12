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

import numpy as np
import pandas as pd
import os
import sys
import argparse
import datetime as dt
from time import time

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

import mnist_helper

parser = argparse.ArgumentParser()
parser.add_argument('--train_size', type = int, default = 600, help = "Size of train dataset.")
parser.add_argument('--test_size', type = int, default = 100, help = "Size of test dataset.")
parser.add_argument('--std_scaler', type = lambda x: (str(x).lower() == 'true'), default = False, help = "Whether to apply standard scaler before PCA.")
parser.add_argument('--pca_percent', type = float, default = .8, help = "How much variance in percent to retain by setting number of components in PCA.")
parser.add_argument('--svm_c', type = float, default = 5.0, help = "Parameter C for svm classifier.")
parser.add_argument('--svm_kernel', default = 'rbf', help = "Kernel used in svm classifier.")
parser.add_argument('--svm_gamma', type = float, default = .05, help = "Parameter gamma for svm classifier.")
parser.add_argument('--lr_solver', default = 'lbfgs', help = "Solver for logistic regression.")
parser.add_argument('--knn_n', type = int, default = 5, help = "Number of neighbors for knn.")
args = parser.parse_args()

# The digits dataset
train_x, train_y, test_x, test_y = mnist_helper.get_dataset('./data/')

if args.std_scaler:
	scaler = StandardScaler()
	train_x = scaler.fit_transform(train_x)
	test_x = scaler.transform(test_x)

pca = PCA(args.pca_percent, whiten = True)
train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x)

print('Shape of train dataset: {}'.format(train_x.shape))
print('Shape of test dataset: {}'.format(test_x.shape))

# Create a classifier: a support vector classifier
print('\nCreate classifier ...\n' + '*' * 50)
classifiers = [
	SVC(C = args.svm_c, kernel = args.svm_kernel, gamma = args.svm_gamma),
	LogisticRegression(solver = args.lr_solver),
	KNeighborsClassifier(n_neighbors = args.knn_n)
]

# Start training
print('\nStart fitting ...\n' + '*' * 50)
for clf in classifiers:
	start_time = dt.datetime.now()
	print('Start {0} classifier training at {1}.'.format(type(clf).__name__, str(start_time)))
	clf.fit(train_x[:args.train_size], train_y[:args.train_size])
	end_time = dt.datetime.now()
	print('End {0} classifier training at {1}.'.format(type(clf).__name__, str(start_time)))
	print('Duration: {0}'.format(str(end_time - start_time)))

# Prediction
print('\nStart prediction ...\n' + '*' * 50)
expected = test_y[:args.test_size]
for clf in classifiers:
	predicted = clf.predict(test_x[:args.test_size])
	print('-' * 50)
	print('Classification report for classifier %s:\n%s\n'
	      % (clf, metrics.classification_report(expected, predicted)))
	print('Accuracy: {0}'.format(metrics.accuracy_score(expected, predicted)))
	print('Confusion matrix:\n%s' % metrics.confusion_matrix(expected, predicted))
