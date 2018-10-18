#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : test_lr
Author          : Charles Young
Python Version  : Python 3.6.1
Requirements    : (Please check document: requirements.txt or use command "pip install -r requirements.txt")
Date            : 2018-10-17
'''

print(__doc__)

import numpy as np
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

import mnist_helper
from classifier.lr import lr

parser = argparse.ArgumentParser()
parser.add_argument('--data_size', type = float, default = .1, help = "Size of data dataset.")
parser.add_argument('--std_scaler', type = lambda x: (str(x).lower() == 'true'), default = False, help = "Whether to apply standard scaler before PCA.")
parser.add_argument('--pca_percent', type = float, default = .8, help = "How much variance in percent to retain by setting number of components in PCA.")
parser.add_argument('--steps', type = int, default = 3000, help = "Maximum number of iter steps.")
parser.add_argument('--lr', type = float, default = 5e-5, help = "Learning rate.")
parser.add_argument('--output', type = lambda x: (str(x).lower() == 'true'), default = False, help = "Whether to print the result report to file.")
parser.add_argument('--outfile', default = './result/report.txt', help = "File to save the result report.")
args = parser.parse_args()

# Get dataset and split into train and test
train_x, train_y, test_x, test_y = mnist_helper.get_dataset('./data/')
train_x, train_y = shuffle(train_x, train_y, random_state = 2333)

# Get part of raw dataset
n_samples = int(args.data_size * train_x.shape[0])
train_x = train_x.values[:n_samples]
train_y = train_y[:n_samples]
n_samples = int(args.data_size * test_x.shape[0])
test_x = test_x.values[:n_samples]
test_y = test_y[:n_samples]

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
clf = lr(num_steps = args.steps, learning_rate = args.lr)

# Start fitting
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
predicted = clf.predict(test_x)
print("""Arguments: 
	data_size: {0},
	std_scaler: {1},
	pca_percent: {2},
	num_steps: {3},
	learning_rate: {4}
	 """.format(args.data_size, args.std_scaler, args.pca_percent, args.steps, args.lr))
print('-' * 50)
print('Classification report for classifier :\n%s\n'
      % (metrics.classification_report(expected, predicted)))
print('Accuracy: {0}'.format(metrics.accuracy_score(expected, predicted)))
print('Confusion matrix:\n%s' % metrics.confusion_matrix(expected, predicted))