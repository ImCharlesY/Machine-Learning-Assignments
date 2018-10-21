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
from datetime import datetime

from sklearn import metrics
from sklearn.utils import shuffle
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from util import mnist_helper
from util import transform

parser = argparse.ArgumentParser()
# parser.add_argument('--train_size', type = float, default = .9, help = "Size of train dataset.")
parser.add_argument('--std_scaler', type = lambda x: (str(x).lower() == 'true'), default = False, help = "Whether to apply standard scaler before PCA.")
parser.add_argument('--pca_percent', type = float, default = .8, help = "How much variance in percent to retain by setting number of components in PCA.")
parser.add_argument('--svm_c', type = float, default = 5.0, help = "Parameter C for svm classifier.")
parser.add_argument('--svm_kernel', default = 'rbf', choices = ['rbf', 'poly', 'rbf'],help = "Kernel used in svm classifier.")
parser.add_argument('--svm_gamma', type = float, default = .05, help = "Parameter gamma for svm classifier.")
parser.add_argument('--svm_degree', type = float, default = 9, help = "Parameter degree for polynomial kernel in svm classifier.")
parser.add_argument('--svm_coef0', type = float, default = 1, help = "Parameter coef0 for polynomial kernel in svm classifier.")
parser.add_argument('--lr_solver', default = 'lbfgs', help = "Solver for logistic regression.")
parser.add_argument('--lr_c', type = float, default = 1.0, help = "Parameter C for svm classifier.")
parser.add_argument('--knn_n', type = int, default = 5, help = "Number of neighbors for knn.")
parser.add_argument('--output', type = lambda x: (str(x).lower() == 'true'), default = False, help = "Whether to print the result report to file.")
parser.add_argument('--outfile', default = './results/report.txt', help = "File to save the result report.")
args = parser.parse_args()

# Get dataset and split into train and test
train_x, train_y, test_x, test_y = mnist_helper.get_dataset('./data/')
train_x, train_y = shuffle(train_x, train_y, random_state = 2333)
# train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, train_size = args.train_size, shuffle = True, random_state = 2333)

# Expand training dataset
train_x, train_y = transform.affine_transform(train_x.values.reshape((len(train_x), 28, 28)), train_y)

# Scaler and decomposition
if args.std_scaler:
	scaler = StandardScaler()
	train_x = scaler.fit_transform(train_x)
	test_x = scaler.transform(test_x)

pca = PCA(args.pca_percent, whiten = True)
train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x)

print('Shape of train dataset: {}'.format(train_x.shape))
print('Shape of test dataset: {}'.format(test_x.shape))

# Create classifiers
print('\nCreate classifier ...\n' + '*' * 50)
classifiers = [
	SVC(C = args.svm_c, kernel = args.svm_kernel, gamma = args.svm_gamma, degree = args.svm_degree, coef0 = args.svm_coef0),
	# LogisticRegression(C = args.lr_c, solver = args.lr_solver, max_iter = 1000),
	# KNeighborsClassifier(n_neighbors = args.knn_n)
]

# Start training
print('\nStart fitting ...\n' + '*' * 50)
for clf in classifiers:
	start_time = dt.datetime.now()
	print('Start {0} classifier training at {1}.'.format(type(clf).__name__, str(start_time)))
	clf.fit(train_x, train_y)
	end_time = dt.datetime.now()
	print('End {0} classifier training at {1}.'.format(type(clf).__name__, str(end_time)))
	print('Duration: {0}'.format(str(end_time - start_time)))

# Prediction
print('\nStart prediction ...\n' + '*' * 50)
expected = test_y
if args.output:
	# Redirect stdout
	if not os.path.exists(os.path.dirname(args.outfile)):
	    os.makedirs(os.path.dirname(args.outfile))
	sys.stdout = open(os.path.splitext(args.outfile)[0] + datetime.now().strftime("_%Y%m%d_%H%M%S") + os.path.splitext(args.outfile)[-1], 'wt')
for clf in classifiers:
	predicted = clf.predict(test_x)
	print("""Parameters: 
		std_scaler: {0},
		pca_percent: {1}
		 """.format(args.std_scaler, args.pca_percent))
	print('-' * 50)
	print('Classification report for classifier %s:\n%s\n'
	      % (clf, metrics.classification_report(expected, predicted)))
	print('Accuracy: {0}'.format(metrics.accuracy_score(expected, predicted)))
	print('Confusion matrix:\n%s' % metrics.confusion_matrix(expected, predicted))
