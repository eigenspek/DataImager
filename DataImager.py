# coding: utf-8

""" #### Developped by Dylan AMELOT, August 1st 2017 

	This little library contains tools to convert a dataset into images """



import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.misc import toimage


def get_ft_weights(X):
	""" Calculates the covariance matrix of Data and then perform EigenDecomposition to get EigenValues corresponding 
		to relative covariance weights assigned to each feature"""
	Xstd = StandardScaler().fit_transform(X)
	cov_matrix = np.cov(Xstd.T) # Covariance Matrix
	e_vals, e_vecs = np.linalg.eig(cov_matrix) # EigenDecomposition
	tot = sum(e_vals)
	var_exp = [(i / tot)*100 for i in e_vals]
	print('relative weights summing-up to %s' % sum(var_exp))
	return var_exp

def transform_weights(l):
	""" turns a df of float weights into a df of int weights for use as image width. """
	l = np.array(l)
	for i in range(len(l)):
		l[i] = [int(round(l[i]))]
		
	return pd.DataFrame(l)


def datatoimg(X, Xt, wmode='1',savemode=False):
	""" Converts a dataset of type DataFrame to an ensemble of images. Please, pass X as type DataFrame. 
	returns two numpy arrays each containing either train or test arrays of corresponding images. 
	Modes : '1' is equal weight for each feature. 'w' each feature will have a width associated to its relative weight."""
	features_number = X.shape[1]
	img_height = getimgheight(X)
	if wmode == '1':
		ximg,xtestimg = subdatatoimg(X,Xt,features_number,img_height,1,savemode)
	elif wmode == 'w':
		ft_weights = get_ft_weights(X)
		ft_weights = pd.DataFrame(ft_weights)
		img_subwidths = transform_weights(ft_weights)
		ximg,xtestimg = subdatatoimg(X,Xt,features_number, img_height,img_subwidths, savemode)
	else : 
		print('###### Error in entered mode, please ensure mode = "1" or "w" ######')
	return np.array(ximg), np.array(xtestimg)

def subdatatoimg(X,Xt, features_number, img_height, widths,savemode):
	""" Actually does the conversion into images, is called by datatoimg(). """
	if savemode:
		os.chdir(os.getcwd() + '/dataimages/train')
	b = X.values
	ximg = []
	ximg = list(ximg)
	xtestimg =[] 
	xtestimg = list(xtestimg)
	for i in range(len(X.values)):
		a = b[i]
#		print(a)
		imgm = imgmatrix(a, features_number, img_height, widths)
		title = i
		ximg.append([imgm])
		if savemode:		
			saveimg(title,imgm)
	print('Converted %d train data to images' % len(b))
	if savemode:
		os.chdir('../')
		os.chdir(os.getcwd() + '/test')

	c = Xt.values
	for i in range(len(Xt.values)):
		d = c[i]
#		print(d)
		imgmt = imgmatrix(d, features_number, img_height, widths)
		title = len(b) + i
		xtestimg.append([imgmt])
		if savemode:
			saveimg(title,imgmt)
	print('Converted %d test data to images' % len(c))
	if savemode:
		os.chdir('../')

	return ximg,xtestimg
#-------------------------------------------
def saveimg(title,image):
	""" function saving images in the dataimages folder"""
	image = toimage(image)
	image.save('%s.png' % str(title))


def getimgheight(X):
	""" searches in dataset all groups in each features and returns max(largest group of each features)  """
	list_features = X.columns
	list_features = list_features.values
	ft_categs = []
	for d in list_features:
		ft_categs.append(max(np.array(X.groupby([d],as_index=False).size().index)))
	print(ft_categs)
	ft_max_categ = max(ft_categs)
	print(ft_max_categ)
	return ft_max_categ + 1

def imgmatrix(X, ft_nbr, height, widths=1):
	""" Here X is one object will all its features. Create a matrix corresponding to the image of this one object"""	
	X = X.T	
	m = np.empty([height,1], dtype=np.int)
	if type(widths) != type(1):
		w = widths.values
		for i in range(ft_nbr):
			a = w[i]
			subm = imgsubmatrix(X[i], height, a)
			m = np.append(m, subm, axis=1)
	else : 
		for i in range(ft_nbr):
			subm = imgsubmatrix(X[i], height, 1)
			m = np.append(m, subm, axis=1)
	m = np.delete(m, 0, 1)	
	return m


def imgsubmatrix(X, height, width=1):
	""" Here X is one feature of the input object and width if !=1 is the associated weight. Creates the submatrix for the 	part of the image associated to this feature """
	X = int(X)
	width = int(width)
	height = int(height)
	subm = np.zeros((height,width), dtype= np.int)
	level = int(X)
	subm[level,:]=1
	return subm
