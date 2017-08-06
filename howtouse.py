# coding: utf-8

import os 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import DataImager as di
from scipy.misc import toimage
from sklearn.preprocessing import StandardScaler

#import data
training_set = pd.read_csv('~/your/path/to/train_set')
testing_set = pd.read_csv('~/your/path/to/test_set')

# prepare data
# - Split features and label to X,y
# - Remove from both train and test sets useless features 

# At this point, your data X and X_test should have:
# - no missing values
# - all features categorized
# - data mapped to integers. ( do it yourself or see sklearn.preprocessing.LabelEncoder() )
# - X and X_test must be of DataFrame type, preferably. 


# Be sure to use a os.chdir() to put the images you will create in a separate folder. 

# ------------------------------------------------------------------------------
# Verbosed way to convert Data to Image with widths = 1 

# Getting the number of features : 
features_number = X.shape[1] 

# couting number of categories per feature and finding the maximum : 
list_features = X.columns
list_features = list_features.values
ft_categs = []
for d in list_features:
	ft_categs.append(max(np.array(X.groupby([d],as_index=False).size().index)))
print('highest categorie in each feature' , ft_categs)
ft_max_categ = max(ft_categs)
print('highest categorie overall features ', ft_max_categ)
img_height = ft_max_categ + 1

# turning only one data object into an image : 
data = X.values
test_image = di.imgmatrix(data[42], features_number, img_height, 1) # we take input object 42. 
print(test_image) # here you can see the associated np.array 
test_img = toimage(test_image)
test_img.save('test_image.png')

#-------------------------------------
# If you want to use relative weights for widths of each feature, define the following 
# and when calling di.imgmatrix replace 1 by img_subwidths. 

# getting relative features weight 
ft_weights = di.get_ft_weights(X)
ft_weights = pd.DataFrame(ft_weights)
# transforming features weight to get img-submatrices width
img_subwidths = di.transform_weights(ft_weights)
# ------------------------------------


# Converting all train data into images and create a list with all images: 
os.chdir(os.getcwd() + '/dataimages/train') # or whatever folder you want. 
b = X.values
ximg = []
ximg = list(ximg)
for i in range(len(X.values)):
	a = b[i]
	print(a)
	imgm = di.imgmatrix(a, features_number, img_height,1)
	ximg.append([imgm])	
	title = i
	img = toimage(imgm)
	img.save('%s.png' % str(title))
os.chdir('../')
Ximg = np.array(ximg) # Here we convert our list into a np.array containing all arrays of each images into one big array. 
print(Ximg[56]) # you can access one image like this

# then you can do the same for test data, be sure to put it in a different folder, and in the loop
# modify title to : title = len(b) + i , in order to name correctly images. 


######################################################################################################################

# Non-Verbosed way to do all of above : 

# you need to provide both the train_set and test_set 

# for equal widths for each feature : 
Xi,Xti = di.datatoimg(X, Xtest,wmode='1',savemode=True) 
print(len(Xi),len(Xti))

# if you want to use weighted widths for each feature : 
Xi,Xti = di.datatoimg(X,Xtest,wmode='w',savemode=True) 
print(len(Xi),len(Xti)) 

# note that Xi,Xti will be returned as numpy arrays. 
print(Xi[42]) # to print image corresponding to object number 42. 

# note : set savemode to False if you do not need to save images and just want to use returned arrays. 




