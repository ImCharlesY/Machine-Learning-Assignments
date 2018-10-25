#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : pca_visual
Author          : Charles Young
Python Version  : Python 3.6.1
Date            : 2018-10-23
'''
print(__doc__)

import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

plt.switch_backend('agg')

sys.path.insert(0,'..')

def draw_vector(v0, v1, ax = None):
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle = '->',
                    linewidth = 2,
                    shrinkA = 0, shrinkB = 0)
    ax.annotate('', v1, v0, arrowprops = arrowprops)

if __name__ == '__main__':

	fig = plt.figure(figsize = (10, 10))

	ax = fig.add_subplot(2, 2, 1)
	# Generate some 2-D points randomly
	rng = np.random.RandomState(0)
	X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
	ax.scatter(X[:, 0], X[:, 1])
	ax.set_title('Original Space')

	ax = fig.add_subplot(2, 2, 2)
	ax.scatter(X[:, 0], np.zeros(len(X[:, 0])), alpha = 0.2)
	ax.scatter(np.zeros(len(X[:, 1])), X[:, 1], alpha = 0.2)
	ax.set_title('Decomposition')

	# Fit a PCA
	pca = PCA(n_components = 2)
	pca.fit(X)

	ax = fig.add_subplot(2, 2, 3)
	# Plot data points with components vectors
	ax.scatter(X[:, 0], X[:, 1], alpha = 0.2)
	for length, vector in zip(pca.explained_variance_, pca.components_):
	    v = vector * 3 * np.sqrt(length)
	    draw_vector(pca.mean_, pca.mean_ + v, ax)
	ax.set_title('Principal Components')

	ax = fig.add_subplot(2, 2, 4)
	# Apply decomposition and reconstruction
	pca = PCA(n_components = 1)
	pca.fit(X)
	X_pca = pca.transform(X)
	X_new = pca.inverse_transform(X_pca)
	ax.scatter(X[:, 0], X[:, 1], alpha = 0.2)
	ax.scatter(X_new[:, 0], X_new[:, 1], alpha = 0.8)
	ax.set_title('Decomposition')

	if not os.path.exists(os.path.join('..', 'figs')):
		os.makedirs(os.path.join('..', 'figs'))
	plt.savefig(os.path.join('..', 'figs', 'pca_visual.svg'), bbox_inches = 'tight', format = 'svg')
