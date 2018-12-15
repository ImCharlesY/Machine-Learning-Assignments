#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : svm
Author          : Charles Young
Python Version  : Python 3.6.2
Date            : 2018-12-01
'''

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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from utils import transforms
from utils.load_dataset import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument('-d', default='./data', nargs='?', help='directory to store dataset. Default \'./data\'')z
parser.add_argument('--pca_percent', type = float, default = .8, nargs='?', help = "How much variance in percent to retain by setting number of components in PCA. Default = 0.8")
parser.add_argument('--svm_c', type = float, default = 5.0, nargs='?', help = "Penalty parameter C of the error term. Default = 5.0")
parser.add_argument('--svm_kernel', default = 'rbf', choices = ['linear', 'poly', 'rbf'], nargs='?', help = "Specifies the kernel type to be used in the algorithm. Default = rbf")
parser.add_argument('--svm_gamma', type = float, default = .025, nargs='?', help = "Kernel coefficient for ‘rbf’ and ‘poly’. Default = 0.025")
parser.add_argument('--svm_degree', type = float, default = 9, nargs='?', help = "Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels. Default = 9")
parser.add_argument('--svm_coef0', type = float, default = 1, nargs='?', help = "Independent term of the polynomial kernel function (‘poly’). Ignored by all other kernels. Default = 1")
parser.add_argument('--output', dest = 'output', action = 'store_const', const = True, help = "Whether to print the result report to file.")
parser.add_argument('--outfile', default = './results/svm/report.txt', nargs='?', help = "File to save the result report. Default = './results/svm/report.txt'")
args = parser.parse_args()

# Get dataset and split into train and test
__, __, train, test = load_dataset(directory=args.d, transform = False)
train_x = train.train_data
train_x = train_x.reshape((train_x.shape[0],-1))
train_y = np.asarray(train.train_labels)
test_x = test.test_data
test_x = test_x.reshape((test_x.shape[0],-1))
test_y = np.asarray(test.test_labels)

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

# Whiten
pca1 = PCA(args.pca_percent, whiten = True)
train_x = pca1.fit_transform(train_x)
test_x = pca1.transform(test_x)

print('Shape of train dataset: {}'.format(train_x.shape))
print('Shape of test dataset: {}'.format(test_x.shape))

# Create classifiers
print('\nCreate classifier ...\n' + '*' * 50)
clf = SVC(C = args.svm_c, kernel = args.svm_kernel, gamma = args.svm_gamma, degree = args.svm_degree, coef0 = args.svm_coef0)

# Start training
print('\nStart fitting ...\n' + '*' * 50)
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
print("""Parameters: 
	pca_percent: {0}
	 """.format(args.pca_percent))
print('-' * 50)
predicted = clf.predict(test_x)
print('Classification report for classifier %s:\n%s\n'
      % (clf, metrics.classification_report(expected, predicted)))
print('Accuracy: {0}'.format(metrics.accuracy_score(expected, predicted)))
print('Confusion matrix:\n{0}'.format(metrics.confusion_matrix(expected, predicted)))
