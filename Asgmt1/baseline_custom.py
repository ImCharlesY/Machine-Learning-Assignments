#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : baseline_custom
Author          : Charles Young
Python Version  : Python 3.6.1
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
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
from classifiers.svm import svm
from classifiers.lr import lr
from classifiers.knn import KNN

from util import mnist_helper

parser = argparse.ArgumentParser()
# parser.add_argument('--train_size', type = float, default = .9, help = "Size of train dataset.")
parser.add_argument('--data_size', type = float, default = .1, nargs='?', help = "Size of data dataset. Default = 0.1")
parser.add_argument('--normalize', dest = 'normal', action = 'store_const', const = True, help = "Whether to normalize the features.")
parser.add_argument('--pca_percent', type = float, default = .8, nargs='?', help = "How much variance in percent to retain by setting number of components in PCA. Default = 0.8")
parser.add_argument('--max_iter', type = int, default = 3000, nargs='?', help = "Hard limit on iterations within solver. Default = 3000")
parser.add_argument('--svm_c', type = float, default = 5.0, nargs='?', help = "Penalty parameter C of the error term. Default = 5.0")
parser.add_argument('--svm_kernel', default = 'rbf', choices = ['linear', 'poly', 'rbf'], nargs='?', help = "Specifies the kernel type to be used in the algorithm. Default = rbf")
parser.add_argument('--svm_gamma', type = float, default = .025, nargs='?', help = "Kernel coefficient for ‘rbf’ and ‘poly’. Default = 0.025")
parser.add_argument('--svm_degree', type = float, default = 9, nargs='?', help = "Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels. Default = 9")
parser.add_argument('--svm_coef0', type = float, default = 1, nargs='?', help = "Independent term of the polynomial kernel function (‘poly’). Ignored by all other kernels. Default = 1")
parser.add_argument('--lr_lr', type = float, default = 5e-5, nargs='?', help = "Learning rate for Logistic Regression. Default = 5e-5")
parser.add_argument('--knn_n', type = int, default = 5, nargs='?', help = "Number of neighbors for knn. Default = 5")
parser.add_argument('--output', dest = 'output', action = 'store_const', const = True, help = "Whether to print the result report to file.")
parser.add_argument('--outfile', default = './results/report.txt', nargs='?', help = "File to save the result report. Default = './results/report.txt'")
args = parser.parse_args()

# Get dataset and split into train and test
train_x, train_y, test_x, test_y = mnist_helper.get_dataset('./data/')
train_x, train_y = shuffle(train_x, train_y, random_state = 2333)
# train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, train_size = args.train_size, shuffle = True, random_state = 2333)

# Get part of raw dataset
n_samples = int(args.data_size * train_x.shape[0])
train_x = train_x.values[:n_samples]
train_y = train_y[:n_samples]
n_samples = int(args.data_size * test_x.shape[0])
test_x = test_x.values[:n_samples]
test_y = test_y[:n_samples]

# Scaler and decomposition
if args.normal:
	scaler = StandardScaler()
	train_x = scaler.fit_transform(train_x)
	test_x = scaler.transform(test_x)

# For svm and lr, with whiten
pca1 = PCA(args.pca_percent, whiten = True)
# pca = PCA(args.pca_percent)
train_x1 = pca1.fit_transform(train_x)
test_x1 = pca1.transform(test_x)

# For knn, without whiten
pca2 = PCA(args.pca_percent)
train_x2 = pca2.fit_transform(train_x)
test_x2 = pca2.transform(test_x)

print('Shape of train dataset: {}'.format(train_x.shape))
print('Shape of test dataset: {}'.format(test_x.shape))

# train_x1 = pd.DataFrame(data = train_x1)
# test_x1 = pd.DataFrame(data = test_x1)
# train_x2 = pd.DataFrame(data = train_x2)
# test_x2 = pd.DataFrame(data = test_x2)

# Create classifiers
print('\nCreate classifier ...\n' + '*' * 50)
classifiers = [
	svm(max_iter = args.max_iter, C = args.svm_c, kernel = args.svm_kernel, gamma = args.svm_gamma, degree = args.svm_degree, coef0 = args.svm_coef0),
	lr(num_steps = args.max_iter, learning_rate = args.lr_lr),
	KNN(n_neighbors = args.knn_n)
]

# Start training
print('\nStart fitting ...\n' + '*' * 50)
for clf in classifiers:
	start_time = dt.datetime.now()
	print('Start {0} classifier training at {1}.'.format(type(clf).__name__, str(start_time)))
	if classifiers.index(clf) == 2:
		clf.fit(train_x2, train_y)
	else:
		clf.fit(train_x1, train_y)
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
print("""Parameters: 
	normalize: {0},
	pca_percent: {1},
	max_iter: {2},
	svm_C: {3},
	svm_kernel: {4},
	svm_gamma: {5},
	svm_degree: {6},
	svm_coef0: {7},
	lr_lr: {8},
	knn_n: {9},
	 """.format(args.normal, args.pca_percent, args.max_iter, args.svm_c, args.svm_kernel, args.svm_gamma, args.svm_degree, args.svm_coef0, args.lr_lr, args.knn_n))
print('-' * 50)
for clf in classifiers:
	if classifiers.index(clf) == 2:
		predicted = clf.predict(test_x2)
	else:
		predicted = clf.predict(test_x1)
	print('Classification report for classifier %s:\n%s\n'
	      % (clf, metrics.classification_report(expected, predicted)))
	print('Accuracy: {0}'.format(metrics.accuracy_score(expected, predicted)))
	print('Confusion matrix:\n%s' % metrics.confusion_matrix(expected, predicted))
