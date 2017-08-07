# coding: utf-8

import os
import numpy as np
import sys
sys.path.append('/home/c3k4/.local/lib/python2.7/site-packages') # just because Pandas was not installed in Neon Env 
sys.path.append('/home/c3k4/Desktop/Programmes/python/DataImager')
import pandas as pd
import DataImager as di

trainset = pd.read_csv('~/Desktop/Programmes/python/titanic/titanic_data/preproc2_train.csv')
testset = pd.read_csv('~/Desktop/Programmes/python/titanic/titanic_data/preproc2_test.csv')

y = trainset['Survived']
trnset = trainset.drop(['PassengerId','Survived'],axis=1)
tstset = testset.drop(['PassengerId'],axis=1)

X,Target = di.datatoimg(trnset,tstset,wmode='1',savemode=False)

traindata = X.reshape(891,144)
target = Target.reshape(418,144)

Xw, Targetw = di.datatoimg(trnset,tstset,wmode='w',savemode=False)

traindataw = Xw.reshape(891,1200)
targetw = Targetw.reshape(418,1200)

os.chdir(os.getcwd() + '/didatas')

np.savetxt('img_train_1.csv', traindata, delimiter=",")
np.savetxt('img_target_1.csv',target,delimiter=",")
np.savetxt('img_train_w.csv',traindataw,delimiter=",")
np.savetxt('img_target_w.csv',targetw,delimiter=",")

train_id_label = pd.DataFrame()
train_id_label['PassengerId'] = trainset['PassengerId']
train_id_label['Survived'] = trainset['Survived']
target_id = testset['PassengerId']
train_id_label.to_csv('train_id_label.csv',index=False)
target_id.to_csv('target_id.csv',index=False)




