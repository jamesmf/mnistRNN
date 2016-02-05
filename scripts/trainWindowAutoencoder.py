'''This is a reproduction of the IRNN experiment
with pixel-by-pixel sequential MNIST in
"A Simple Way to Initialize Recurrent Networks of Rectified Linear Units"
by Quoc V. Le, Navdeep Jaitly, Geoffrey E. Hinton

arXiv:1504.00941v2 [cs.NE] 7 Apr 201
http://arxiv.org/pdf/1504.00941v2.pdf

Optimizer is replaced with RMSprop which yields more stable and steady
improvement.

Reaches 0.93 train/test accuracy after 900 epochs
(which roughly corresponds to 1687500 steps in the original paper.)
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.utils import np_utils

import json


def im2Window(image,wSize):
    xdim    = image.shape[1] - wSize + 1
    ydim    = image.shape[0] - wSize + 1
    #numWins = xdim*ydim
    output  = []
    [ output.append(image[y:y+wSize,x:x+wSize])  for y in range(0,ydim) for x in range(0,xdim)]
    return np.array(output)

batch_size      = 32
nb_classes      = 10
nb_epochs       = 2
hidden_units    = 100
repSize         = 20
wSize           = 15

learning_rate   = 1e-6
clip_norm       = 1.0

# the data, shuffled and split between train and test sets
(X_train_raw, y_train), (X_test_raw, y_test) = mnist.load_data()

print("X_train_raw shape: ", X_train_raw.shape)
print("X_test_raw shape: ", X_test_raw.shape)
del y_train
del y_test

X_train  = []
X_test   = []
[X_train.append(im2Window(image,wSize)) for image in X_train_raw]
[X_test.append(im2Window(image,wSize)) for image in X_test_raw]

del X_train_raw
del X_test_raw

Xtrain2 = []
ytrain  = []
Xtest2  = []
ytest   = []
for i in range(0,len(X_train)):
    for j in range(0,len(X_train[i])):
        Xtrain2.append(X_train[i][j])
        ytrain.append(np.ndarray.flatten(X_train[i][j]))
        if i < len(X_test):
            Xtest2.append(X_test[i][j])
            ytest.append(np.ndarray.flatten(X_test[i][j]))
        
X_train     = np.array(Xtrain2)
X_test      = np.array(Xtest2)
ytest       = np.array(ytest)
ytrain      = np.array(ytrain)
#X_train = X_train.reshape(X_train.shape[0], -1, 1)
#X_test = X_test.reshape(X_test.shape[0], -1, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
#Y_train = np_utils.to_categorical(y_train, nb_classes)
#Y_test = np_utils.to_categorical(y_test, nb_classes)

inshape     = X_train.shape[1:]
outshape    = X_train.shape[1]

print('Evaluate IRNN...')
model = Sequential()


model.add(Convolution2D(8,4, 4, input_shape=(1, wSize, wSize))) 
model.add(Activation('relu'))

model.add(Convolution2D(16, 4, 4)) 
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(repSize))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(wSize**2, init='normal'))

rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='mean_squared_error', optimizer=rmsprop)

model.fit(X_train, ytrain, batch_size=batch_size, nb_epoch=nb_epochs,
          show_accuracy=True, verbose=1, validation_data=(X_test, X_test))

scores = model.evaluate(X_test, ytest, show_accuracy=True, verbose=0)
print('IRNN test score:', scores[0])
print('IRNN test accuracy:', scores[1])

del X_train

model2 = Sequential()
model2.add(Convolution2D(8,4, 4, input_shape=(1, wSize, wSize))) 
model2.add(Activation('relu'))

model2.add(Convolution2D(16, 4, 4)) 
model2.add(Activation('relu'))

model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))

model2.add(Flatten())
model2.add(Dense(repSize))
rmsprop = RMSprop(lr=learning_rate)
model2.compile(loss='mean_squared_error', optimizer=rmsprop)

for layernum in range(0,len(model2.layers)):
    model2.layers[layernum].set_weights(model.layers[layernum].get_weights())    
    
jsonstring  = model2.to_json()
with open("../autoEncoder.json",'wb') as f:
    f.write(jsonstring)
model2.save_weights("../autoEncoder.h5")



#
#print('Compare to LSTM...')
#model = Sequential()
#model.add(LSTM(hidden_units, input_shape=X_train.shape[1:]))
#model.add(Dense(nb_classes))
#model.add(Activation('softmax'))
#rmsprop = RMSprop(lr=learning_rate)
#model.compile(loss='categorical_crossentropy', optimizer=rmsprop)
#
#model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
#          show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
#
#scores = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
#print('LSTM test score:', scores[0])
#print('LSTM test accuracy:', scores[1])
