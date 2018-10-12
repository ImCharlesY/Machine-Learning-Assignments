#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : mnist_helper
Author          : Charles Young
Python Version  : Python 3.6.1
Requirements    : (Please check document: requirements.txt or use command "pip install -r requirements.txt")
Date            : 2018-10-06
'''
import gzip
import numpy as np
import pandas as pd
import os
import shutil
import struct
import sys

try:
	from urllib.request import urlretrieve
except ImportError:
	from urllib import urlretrieve

def get_dataset(gzfdir):
# Functions to load MNIST images and unpack into train and test set.
# - loadData reads a image and formats it into a 28x28 long array
# - loadLabels reads the corresponding label data, one for each image
# - load packs the downloaded image and label data into a combined format to be read later by
#   the CNTK text reader
	def loadData(src, cimg, gzfdir):
		gzfname = os.path.join(gzfdir, os.path.basename(src))
		if os.path.exists(gzfname):
			print (src + ' already exists! ')
		else:
			print ('Downloading ' + src)
			urlretrieve(src, gzfname)
			print ('Done.')
		try:
			with gzip.open(gzfname) as gz:
				n = struct.unpack('I', gz.read(4))
				# Read magic number.
				if n[0] != 0x3080000:
					raise Exception('Invalid file: unexpected magic number.')
				# Read number of entries.
				n = struct.unpack('>I', gz.read(4))[0]
				if n != cimg:
					raise Exception('Invalid file: expected {0} entries.'.format(cimg))
				crow = struct.unpack('>I', gz.read(4))[0]
				ccol = struct.unpack('>I', gz.read(4))[0]
				if crow != 28 or ccol != 28:
					raise Exception('Invalid file: expected 28 rows/cols per image.')
				# Read data.
				res = np.fromstring(gz.read(cimg * crow * ccol), dtype = np.uint8)
		finally:
			None
			# os.remove(gzfname)
		return res.reshape((cimg, crow * ccol))

	def loadLabels(src, cimg, gzfname):
		gzfname = os.path.join(gzfdir, os.path.basename(src))
		if os.path.exists(gzfname):
			print (src + ' already exists! ')
		else:
			print ('Downloading ' + src)
			urlretrieve(src, gzfname)
			print ('Done.')
		try:
			with gzip.open(gzfname) as gz:
				n = struct.unpack('I', gz.read(4))
				# Read magic number.
				if n[0] != 0x1080000:
					raise Exception('Invalid file: unexpected magic number.')
				# Read number of entries.
				n = struct.unpack('>I', gz.read(4))
				if n[0] != cimg:
					raise Exception('Invalid file: expected {0} rows.'.format(cimg))
				# Read labels.
				res = np.fromstring(gz.read(cimg), dtype = np.uint8)
		finally:
			None
			# os.remove(gzfname)
		return res.reshape((cimg, ))

	# URLs for the train image and label data
	url_train_image = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
	url_train_labels = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
	num_train_samples = 60000
	print("Downloading train data ...")
	train_data = pd.DataFrame(data = loadData(url_train_image, num_train_samples, gzfdir))
	train_labels = loadLabels(url_train_labels, num_train_samples, gzfdir)

	# URLs for the test image and label data
	url_test_image = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
	url_test_labels = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
	num_test_samples = 10000
	print("Downloading test data ...")
	test_data = pd.DataFrame(data = loadData(url_test_image, num_test_samples, gzfdir))
	test_labels = loadLabels(url_test_labels, num_test_samples, gzfdir)
	return pd.concat([train_data, test_data], ignore_index = True), np.hstack((train_labels, test_labels))
