# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 21:45:11 2018

@author: ponkotu_androido
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import cv2
import glob
import random
random.seed(0)

from keras.models import Sequential  
from keras.layers.core import Dense, Dropout
from keras.optimizers import RMSprop

from make_model import mk_model

size = 64
idol = {1:'honda_mio',
        2:'tada_riina'}

def train_test_split(df_x, df_y, test_size=0.1):  
    """
    This just splits data to training and testing parts
    """
    random.shuffle(df_x)
    random.shuffle(df_y)

    ntrn = round(len(df_y) * (1 - test_size))
    ntrn = int(ntrn)
    X_train = df_x[0:ntrn]
    Y_train = df_y[0:ntrn]
    X_test = df_x[ntrn:]
    Y_test = df_y[ntrn:]
    return (X_train, Y_train), (X_test, Y_test)

#%%画像読み込み、学習、教師データ生成

data_X = []
data_Y = []
for i in idol.keys():
    print(idol[i])
    for f in glob.glob('re_data/faces/'+idol[i]+'/*.jpg'):
        image = cv2.imread(f)
        gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image,(size,size))
        Y = np.zeros((len(idol)))
        data_X.append(image)
        Y[i-1] = 1
        data_Y.append(Y)
data_X = np.array(data_X)
data_Y = np.array(data_Y)
data_X = data_X.reshape(-1, size**2*3).astype('float32') / 255
(X_train, Y_train), (X_test, Y_test) = train_test_split(data_X, data_Y)

#%%
inp_size = size**2*3
out_size = 2

model = mk_model(inp_size, out_size)
#%%
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

hist = model.fit(X_train, Y_train,
                 batch_size=5,
                 epochs=20,
                 verbose=1,
                 validation_data=(X_test, Y_test))
#%%

score = model.evaluate(X_test, Y_test, verbose=1)
print('succses = ', score[1], ' loss=', score[0])

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_loss'])
plt.title('loss')
plt.show()

#%%
im = cv2.imread('re_data/faces/tada_riina/2303502.jpg')
im = cv2.resize(im, (size, size))
plt.imshow(im)
plt.show()

#%%
im = im.reshape(-1, size**2*3).astype('float32') / 255

r = model.predict(np.array(im), batch_size=3, verbose=1)
res = r[0]

for i, acc in enumerate(res):
    print(idol[i+1], '=', int(acc * 100))
print('result---', idol[res.argmax()+1])
#%%
model.save_weights('result/model/derepa001.h5')



