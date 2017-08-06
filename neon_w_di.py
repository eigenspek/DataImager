# coding: utf-8

#!/usr/bin/env python
import os
import numpy as np
import sys
sys.path.append('/home/c3k4/.local/lib/python2.7/site-packages') # just because Pandas was not installed in Neon Env 
sys.path.append('/home/c3k4/Desktop/Programmes/python/DataImager')
sys.path.append('/usr/local/lib/python2.7/dist-packages') # for SciKit Learn 
import pandas as pd 
import DataImager as di


trainset = pd.read_csv('~/Desktop/Programmes/python/titanic/titanic_data/preproc2_train.csv')
testset = pd.read_csv('~/Desktop/Programmes/python/titanic/titanic_data/preproc2_test.csv')

y = trainset['Survived']
trainset = trainset.drop(['PassengerId','Survived'],axis=1)
testset = testset.drop(['PassengerId'],axis=1)

X,Target = di.datatoimg(trainset,testset,wmode='1',savemode=False)

X_train = X[0:800]
y_train = np.array(y[0:800])
X_test = X[801:891]
y_test = np.array(y[801:891])

# Now we need to work a bit on the shape of our datasets, for images, in order to create an array iterator, we need
# to shape X like : (#examples,#features) where features should be in order : (channels, height, width). 
# Also, since ArrayIterator uses batch-size determined by the backend, be sure to generate backend first. 

#Since our number of channels is 1, no need to overthink it: 
Xtrain = X_train.reshape(800,144)
Xtest = X_test.reshape(90,144)

print(Xtrain.shape, y_train.shape)
print(Xtest.shape, y_test.shape)
# Now we generate our backend, we may need to think again the batchsize : 
from neon.backends import gen_backend
be = gen_backend(backend='cpu',batch_size=64)
print be
# and now we need to create the iterator : 
from neon.data import ArrayIterator
train_set = ArrayIterator(Xtrain, y_train, nclass=2, lshape=(1,12,12))
test_set = ArrayIterator(Xtest, y_test, nclass=2, lshape=(1,12,12))


from neon.layers import Conv, Affine, Pooling, Dropout
from neon.initializers import Uniform, Constant, Gaussian
from neon.transforms.activation import Rectlin, Softmax

init_uni = Uniform(low=-0.1, high=0.1) # try Gaussian(loc=0, scale=0.1)
init_cst = Constant(0.1) #to avoid dead neurons 
layers = []

layers.append(Conv(fshape=(2,2,32), init=init_uni, bias=init_cst, padding=0, activation=Rectlin()))
layers.append(Pooling(fshape=2, strides=2))
layers.append(Affine(nout=2, init=init_uni, activation=Softmax()))

from neon.models import Model
model = Model(layers)

from neon.layers import GeneralizedCost
from neon.transforms import CrossEntropyBinary
cost = GeneralizedCost(costfunc=CrossEntropyBinary())

from neon.optimizers import GradientDescentMomentum
optimizer = GradientDescentMomentum(0.1, momentum_coef=0.9)

from neon.callbacks.callbacks import Callbacks
callbacks = Callbacks(model, train_set)

model.fit(dataset=train_set, cost=cost, optimizer=optimizer,  num_epochs=10, callbacks=callbacks)

from neon.transforms import Misclassification
error_pct = 100 * model.eval(test_set, metric=Misclassification())
accuracy_fp = 100 - error_pct
print 'Model Accuracy : %.1f%%' % accuracy_fp 
