'''
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop
import scipy.misc as mi

batch_size      = 32
nb_classes      = 10
nb_epochs       = 50
repSize         = 200
wSize           = 28

learning_rate   = 0.001
clip_norm       = 1.0

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#cutoff  = 10000
#X_train = X_train[0:cutoff,:,:]
#X_test  = X_test[0:cutoff,:,:]
X_train = np.reshape(X_train,(X_train.shape[0],1,wSize,wSize))
X_test  = np.reshape(X_test,(X_test.shape[0],1,wSize,wSize))



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
print('X_train shape:', X_train.shape)
print('X_train shape:', X_test.shape)
print('ytrain shape:', ytest.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
#Y_train = np_utils.to_categorical(y_train, nb_classes)
#Y_test = np_utils.to_categorical(y_test, nb_classes)

inshape     = X_train.shape[1:]
outshape    = X_train.shape[1]

print("defining model")
model = Sequential()
model.add(Convolution2D(8,4, 4, input_shape=(1, wSize, wSize))) 
model.add(Activation('relu'))

model.add(Convolution2D(8, 4, 4)) 
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(16, 4, 4)) 
model.add(Activation('relu'))

model.add(Convolution2D(16, 4, 4)) 
model.add(Activation('relu'))

model.add(Convolution2D(16, 4, 4)) 
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(repSize))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(wSize**2, init='normal'))

rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='mean_squared_error', optimizer=rmsprop)
print("model compiled")

model.fit(X_train, ytrain, batch_size=batch_size, nb_epoch=nb_epochs,
          show_accuracy=True, verbose=1, validation_data=(X_test, ytest))


jsonstring  = model.to_json()
with open("../models/autoEncoderFull.json",'wb') as f:
    f.write(jsonstring)
model.save_weights("../models/autoEncoderFull.h5",overwrite=True)


model2 = Sequential()
model2.add(Convolution2D(8,4, 4, input_shape=(1, wSize, wSize))) 
model2.add(Activation('relu'))

model2.add(Convolution2D(8, 4, 4)) 
model2.add(Activation('relu'))

model2.add(MaxPooling2D(pool_size=(2, 2)))

model2.add(Convolution2D(16, 4, 4)) 
model2.add(Activation('relu'))

model2.add(Convolution2D(16, 4, 4)) 
model2.add(Activation('relu'))

model2.add(Convolution2D(16, 4, 4)) 
model2.add(Activation('relu'))



model2.add(Flatten())
model2.add(Dense(repSize))
rmsprop = RMSprop(lr=learning_rate)
model2.compile(loss='mean_squared_error', optimizer=rmsprop)

for layernum in range(0,len(model2.layers)):
    model2.layers[layernum].set_weights(model.layers[layernum].get_weights())    
    
jsonstring  = model2.to_json()
with open("../models/autoEncoder.json",'wb') as f:
    f.write(jsonstring)
model2.save_weights("../models/autoEncoder.h5",overwrite=True)

del model2

for num, pred in enumerate(model.predict(X_test)):
    print(pred.shape)
    mi.imsave("before.jpg",X_test[num][0][:,:])
    mi.imsave("after.jpg",np.reshape(pred,(28,28)))
    stop    = raw_input("")