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
#np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
#from keras.initializations import normal, identity
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.optimizers import RMSprop, Adadelta
#from keras.utils import np_utils

from keras.models import model_from_json
#import json


#def im2Window(image,wSize):
#    xdim    = image.shape[1] - wSize + 1
#    ydim    = image.shape[0] - wSize + 1
#    #numWins = xdim*ydim
#    output  = []
#    [ output.append(np.ndarray.flatten(image[y:y+wSize,x:x+wSize]))  for y in range(0,ydim) for x in range(0,xdim)]
#    return np.array(output)

#def autoEncode(image,wSize,model):
#    xdim    = image.shape[1] - wSize + 1
#    ydim    = image.shape[0] - wSize + 1
#    #numWins = xdim*ydim
#    output  = []
#    #print(image[0:0+wSize,0:0+wSize])
#    #temp    = np.reshape(np.ndarray.flatten(image[0:0+wSize,0:0+wSize]),(1,wSize**2))
#    
#    #print(model.predict(temp))
#    #print("I did it")
#    [ output.append(model.predict(np.reshape(np.ndarray.flatten(image[y:y+wSize,x:x+wSize]),(1,wSize**2)))[0])  for y in range(0,ydim) for x in range(0,xdim)]
#    return np.array(output)
    
def autoencode(images,model):
    images  = np.reshape(images,(images.shape[0],1,images.shape[1],images.shape[2]))
    return model.predict(images) 
    
def loadThatModel(folder):
    with open(folder+".json",'rb') as f:
        #json_string     = json.load(f)
        json_string     = f.read()
    model = model_from_json(json_string)
    model.load_weights(folder+".h5")
    return model

#define some run parameters
batch_size      = 32
nb_epochs       = 10
examplesPer     = 1000
maxToAdd        = 5
hidden_units    = 100
wSize           = 10

learning_rate   = 1e-6
clip_norm       = 1.0
cutoff          = 1000



# the data, shuffled and split between train and test sets
(X_train_raw, y_train_temp), (X_test_raw, y_test_temp) = mnist.load_data()

#ignore "cutoff" section in full run
X_train_raw     = X_train_raw[:cutoff]
X_test_raw      = X_test_raw[:cutoff]
y_train_temp    = y_train_temp[:cutoff]
y_test_temp     = y_test_temp[:cutoff]

#basic image processing
X_train_raw     = X_train_raw.astype('float32')
X_test_raw      = X_test_raw.astype('float32')
X_train_raw     /= 255
X_test_raw      /= 255


#load our autoencoder, trained previously
autoencoder     = loadThatModel("../models/autoEncoder")

#autoencode our images to vectors
X_train_vecs  = autoencode(X_train_raw,autoencoder)
X_test_vecs   = autoencode(X_test_raw,autoencoder)




print('X_train shape:', X_train_vecs.shape)
print(X_train_vecs.shape[0], 'train samples')
print(X_test_vecs.shape[0], 'test samples')


#define our RNN setup
model = Sequential()

model.add(GRU(output_dim=hidden_units,input_shape = (maxToAdd,len(X_train_vecs[0]))))

model.add(Dense(1))
model.add(Activation('relu'))

optimizer   = Adadelta()
model.compile(loss='mean_squared_error', optimizer=optimizer)





#run epochs of sampling data then training
for ep in range(0,nb_epochs):
    X_train       = []
    y_train       = []
    X_test        = []
    y_test        = []    
    
    X_train     = np.zeros((examplesPer,maxToAdd,len(X_train_vecs[0])))
    for i in range(0,examplesPer):
        output      = np.zeros((maxToAdd,len(X_train_vecs[0])))
        #print(output.shape)
        numToAdd    = np.ceil(np.random.rand()*maxToAdd)
        indices     = np.random.choice(X_train_vecs.shape[0],size=numToAdd)
        #print(indices)
        example     = X_train_vecs[indices]
        #print("example shape: ", example.shape)
        exampleY    = y_train_temp[indices]
        #print(example.shape, exampleY)
        output[0:numToAdd,:] = example
        X_train[i,:,:] = output
        y_train.append(np.sum(exampleY))
    
    
    X_train     = np.array(X_train)
    y_train     = np.array(y_train)
    
    
    X_test     = np.zeros((examplesPer,maxToAdd,len(X_test_vecs[0])))
    for i in range(0,examplesPer):
        output      = np.zeros((maxToAdd,len(X_test_vecs[0])))
        #print(output.shape)
        numToAdd    = np.ceil(np.random.rand()*maxToAdd)
        indices     = np.random.choice(X_test_vecs.shape[0],size=numToAdd)
        #print(indices)
        example     = X_test_vecs[indices]
        #print("example shape: ", example.shape)
        exampleY    = y_test_temp[indices]
        #print(example.shape, exampleY)
        output[0:numToAdd,:] = example
        X_test[i,:,:] = output
        y_test.append(np.sum(exampleY))

    X_test  = np.array(X_test)
    y_test  = np.array(y_test)    
    
    
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1,
              verbose=1)
    

preds   = model.predict(X_test)

for num, ans in enumerate(y_test):
    print("predicted: ", preds[num], "actual: ", ans)
    
print(np.sum(np.sqrt([ (y_test[i] - preds[i][0])**2 for i in range(0,len(preds)) ])))
print("naive guess", np.sum(np.sqrt([ (y_test[i] - np.mean(y_test))**2 for i in range(0,len(y_test)) ])))






jsonstring  = model.to_json()
with open("../models/basicRNN.json",'wb') as f:
    f.write(jsonstring)
model.save_weights("../models/basicRNN.h5",overwrite=True)

