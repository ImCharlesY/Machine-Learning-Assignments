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

import mnist_helper

parser = argparse.ArgumentParser()
# parser.add_argument('--train_size', type = float, default = .9, help = "Size of train dataset.")
parser.add_argument('--third_clf', type = int, default = 2, choices = [0, 1, 2], help = "Specify which classifier to be the final classifier (0 - svm; 1 - lr; 2 - knn).")
parser.add_argument('--std_scaler', type = lambda x: (str(x).lower() == 'true'), default = False, help = "Whether to apply standard scaler before PCA.")
parser.add_argument('--pca_percent', type = float, default = .8, help = "How much variance in percent to retain by setting number of components in PCA.")
parser.add_argument('--svm_c', type = float, default = 5.0, help = "Parameter C for svm classifier.")
parser.add_argument('--svm_kernel', default = 'rbf', help = "Kernel used in svm classifier.")
parser.add_argument('--svm_gamma', type = float, default = .05, help = "Parameter gamma for svm classifier.")
parser.add_argument('--lr_solver', default = 'lbfgs', help = "Solver for logistic regression.")
parser.add_argument('--knn_n', type = int, default = 5, help = "Number of neighbors for knn.")
parser.add_argument('--output', type = lambda x: (str(x).lower() == 'true'), default = False, help = "Whether to print the result report to file.")
parser.add_argument('--outfile', default = './result/report.txt', help = "File to save the result report.")
args = parser.parse_args()

# Get dataset and split into train and test
train_x, train_y, test_x, test_y = mnist_helper.get_dataset('./data/')
train_x, train_y = shuffle(train_x, train_y, random_state = 2333)
# train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, train_size = args.train_size, shuffle = True, random_state = 2333)

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

train_x = pd.DataFrame(data = train_x)
test_x = pd.DataFrame(data = test_x)


def stacking_train(clf_a, clf_b):

	''' Slice dataset into 5 folds '''
	def cutData(data_x, data_y):
		step = data_x.shape[0]//5
		slice_x = list(data_x[i*step:(i+1)*step] for i in range(4))
		slice_x.append(data_x[4*step:])
		slice_y = list(data_y[i*step:(i+1)*step] for i in range(4))
		slice_y.append(data_y[4*step:])
		return slice_x, slice_y

	print('\nSlice dataset into 5 folds ...\n' + '-' * 50)
	slice_x, slice_y = cutData(train_x, train_y)

	stackls_a = []
	stackls_b = []

	''' Train stacking '''
	for num in range(5):
		print('\nBegin stacking {0} ...\n'.format(num + 1) + '-' * 50)
		st_train_x = pd.concat([slice_x[i] for i in range(5) if i != num], ignore_index = True)
		st_train_y = np.hstack(tuple([slice_y[i] for i in range(5) if i != num]))

		clf_a.fit(st_train_x, st_train_y)
		clf_b.fit(st_train_x, st_train_y)
		stackls_a += clf_a.predict(slice_x[num]).tolist()
		stackls_b += clf_b.predict(slice_x[num]).tolist()

		print('Current length of stacking: {0}'.format(len(stackls_a)))

	print('Concat stacking...')
	stack_feat = pd.DataFrame({'stack_' + type(clf_a).__name__:stackls_a, 'stack_' + type(clf_b).__name__:stackls_b})
	return pd.concat([train_x, stack_feat], axis = 1)


def stacking_test(clf_a, clf_b):

	clf_a.fit(train_x, train_y)
	clf_b.fit(train_x, train_y)
	stackls_a = clf_a.predict(test_x)
	stackls_b = clf_b.predict(test_x)

	print('Concat stacking...')
	stack_feat = pd.DataFrame({'stack_' + type(clf_a).__name__:stackls_a, 'stack_' + type(clf_b).__name__:stackls_b})
	return pd.concat([test_x, stack_feat], axis = 1)


# Create classifiers
print('\nCreate classifier ...\n' + '*' * 50)
classifiers = [
	SVC(C = args.svm_c, kernel = args.svm_kernel, gamma = args.svm_gamma),
	LogisticRegression(solver = args.lr_solver),
	KNeighborsClassifier(n_neighbors = args.knn_n)
]

# Choose 2 classifiers to generate stacking
clf_stack = [clf for clf in range(3) if clf is not args.third_clf]

# Start learning stacking
print('\nStart learning stacking using {0} and {1} ...\n'.format(type(classifiers[clf_stack[0]]).__name__, type(classifiers[clf_stack[1]]).__name__) + '*' * 50)
start_time = dt.datetime.now()
print('Start learning training dataset stacking at {0}.'.format(str(start_time)))
train_x_stack = stacking_train(classifiers[clf_stack[0]], classifiers[clf_stack[1]])
start_time = dt.datetime.now()
print('Start learning test dataset stacking at {0}.'.format(str(start_time)))
test_x_stack = stacking_test(classifiers[clf_stack[0]], classifiers[clf_stack[1]])

# Start fitting training dataset with the third classifier
start_time = dt.datetime.now()
print('Startfitting training dataset with the third classifier {0} at {1}.'.format(type(classifiers[args.third_clf]).__name__, str(start_time)))
classifiers[args.third_clf].fit(train_x_stack, train_y)
end_time = dt.datetime.now()
print('End fitting at {0}.'.format(str(end_time)))
print('Duration: {0}'.format(str(end_time - start_time)))

# Prediction
print('\nStart prediction ...\n' + '*' * 50)
expected = test_y
if args.output:
	# Redirect stdout
	if not os.path.exists(os.path.dirname(args.outfile)):
	    os.makedirs(os.path.dirname(args.outfile))
	sys.stdout = open(os.path.splitext(args.outfile)[0] + datetime.now().strftime("_%Y%m%d_%H%M%S") + os.path.splitext(args.outfile)[-1], 'wt')
predicted = classifiers[args.third_clf].predict(test_x_stack)
print("""Arguments: 
	third_clf: {0},
	std_scaler: {1},
	pca_percent: {2},
	svm_c: {3},
	svm_kernel: {4},
	svm_gamma: {5},
	lr_solver: {6},
	knn_n: {7} """.format(type(classifiers[args.third_clf]).__name__, args.std_scaler, args.pca_percent, args.svm_c, args.svm_kernel, args.svm_gamma, args.lr_solver, args.knn_n))
print('-' * 50)
print('Classification report for classifier %s:\n%s\n'
      % (classifiers[args.third_clf], metrics.classification_report(expected, predicted)))
print('Accuracy: {0}'.format(metrics.accuracy_score(expected, predicted)))
print('Confusion matrix:\n%s' % metrics.confusion_matrix(expected, predicted))
