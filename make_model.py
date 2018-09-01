# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 01:04:07 2018

@author: ponkotu_androido
"""
from keras.models import Sequential  
from keras.layers import Dense, Dropout, Conv2D,MaxPooling2D,Activation,Flatten
from keras.optimizers import RMSprop

def mk_model(inp, out):
    '''
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(inp,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', input_shape=(inp,)))
    model.add(Dropout(0.2))
    model.add(Dense(out, activation='softmax'))
    '''
    
    model = Sequential()
    model.add(Conv2D(input_shape=(inp, inp, 3), filters=32,kernel_size=(2, 2), strides=(1, 1), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dense(out))
    model.add(Activation('sigmoid'))
    
    
    
    return model