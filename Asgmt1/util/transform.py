#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : transform
Author          : Charles Young
Python Version  : Python 3.6.1
Date            : 2018-10-23
'''

import cv2  
import numpy as np  

# class for calculation of the Histogram of Oriented Gradients (HOG) descriptor
class HOG:

	def __init__(self):
		winSize = (20,20)
		blockSize = (10,10)
		blockStride = (5,5)
		cellSize = (10,10)
		nbins = 9
		derivAperture = 1
		winSigma = -1.
		histogramNormType = 0
		L2HysThreshold = 0.2
		gammaCorrection = 1
		nlevels = 64
		useSignedGradients = True
		 
		self.hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, useSignedGradients)
	
	def compute(self, im):
		return np.ravel(self.hog.compute(im))

def gethog(ims):
	# Calc hog of a list of images.
	# paras: 
	#	ims -- np.ndarray: input images
	# return: 
	#	descriptor -- np.ndarray: the Histogram of Oriented Gradients (HOG) descriptor with the size of (81,)
	h = HOG()
	descriptor = []
	for im in ims:
		descriptor.append(np.ravel(h.compute(im.reshape(28,28))))
	return np.array(descriptor)	

def deskew(ims):
	# Apply deskewing to a list of images.
	# paras: 
	#	ims -- np.ndarray: input images
	# return:
	#	ims -- np.ndarray: images after deskewing
	def _deskew(img):
		SZ = 28
		m = cv2.moments(img)
		if abs(m['mu02']) < 1e-2:
			# no deskewing needed. 
			return img.copy()
		# Calculate skew based on central momemts. 
		skew = m['mu11']/m['mu02']
		# Calculate affine transform to correct skewness. 
		M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
		# Apply affine transform
		img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
		return img
	for idx in range(len(ims)):
		ims[idx] = _deskew(ims[idx].reshape(28,28)).reshape(28*28,)
	return ims
