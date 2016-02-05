# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 23:10:43 2016

@author: jmf
"""
from __future__ import print_function
from keras.datasets import mnist
import scipy.misc as mi
from keras.models import model_from_json

import numpy as np

def loadThatModel(folder):
    with open(folder+".json",'rb') as f:
        #json_string     = json.load(f)
        json_string     = f.read()
    model = model_from_json(json_string)
    model.load_weights(folder+".h5")
#    for i in range(0,len(model.layers)):
#        print(model.get_weights()[i].shape,end="")
#        print(" "+str(i))
    return model

model     = loadThatModel("../models/autoEncoderFull")

(X_train, y_train), (X_test, y_test) = mnist.load_data()

#cutoff  = 10000
#X_train = X_train[0:cutoff,:,:]
#X_test  = X_test[0:cutoff,:,:]
X_train = np.reshape(X_train,(X_train.shape[0],1,28,28))
X_test  = np.reshape(X_test,(X_test.shape[0],1,28,28))



ytrain  = np.array([np.ndarray.flatten(im[0]) for im in X_train])
ytest   = np.array([np.ndarray.flatten(im[0]) for im in X_test])

del y_train
del y_test


#X_train = X_train.reshape(X_train.shape[0], -1, 1)
#X_test = X_test.reshape(X_test.shape[0], -1, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

preds   = model.predict(X_test)
print(np.mean(preds))

for num, pred in enumerate(preds):
    print(pred)
    print(num,np.mean(pred))
    mi.imsave("before.jpg",X_test[num][0][:,:])
    mi.imsave("after.jpg",np.reshape(pred,(28,28)))
    stop = raw_input("")