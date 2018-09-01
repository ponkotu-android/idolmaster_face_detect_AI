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
import os
from PIL import Image, ImageDraw, ImageFont
random.seed(0)

import keras
from keras.models import Sequential  
from keras.layers import Dense, Dropout,Conv2D,MaxPooling2D,Activation,Flatten
from keras.optimizers import RMSprop

size = 100
idol = {1:'honda_mio',
        2:'tada_riina'}

def train_test_split(df_x, df_y, test_size=0.1):  
    """
    This just splits data to training and testing parts
    """
    i = list(range(len(df_y)))
    random.shuffle(i)

    ntrn = round(len(df_y) * (1 - test_size))
    ntrn = int(ntrn)
    X_train = df_x[i[0:ntrn]]
    Y_train = df_y[i[0:ntrn]]
    X_test = df_x[i[ntrn:]]
    Y_test = df_y[i[ntrn:]]
    return (X_train, Y_train), (X_test, Y_test)

# %% 画像読み込み、学習、教師データ生成
data_X = []
data_Y = []
for i in idol.keys():
    print(idol[i])
    for f in glob.glob('re_data/faces3/'+idol[i]+'/*'):
        image = cv2.imread(f)
        #gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image,(size,size))
        data_X.append(image)
        data_Y.append(i-1)
data_X = np.array(data_X)
data_Y = np.array(data_Y)
data_X = data_X.astype('float32') / 255
(X_train, Y_train), (X_test, Y_test) = train_test_split(data_X, data_Y)
Y_train = keras.utils.np_utils.to_categorical(Y_train.astype('int32'),len(idol))
Y_test = keras.utils.np_utils.to_categorical(Y_test.astype('int32'),len(idol))
print('train_data='+str(len(Y_train)))
print('test_data ='+str(len(Y_test)))

#%% モデル定義、学習,保存
inp_size = size
out_size = len(idol)
model = mk_model(inp_size, out_size)
model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

hist = model.fit(X_train, Y_train, batch_size=10,
                 epochs=60, verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=1)
print('succses = ', score[1], ' loss=', score[0])

plt.plot(hist.history['acc'], label = 'acc')
plt.plot(hist.history['val_loss'], label = 'val_loss')
plt.legend()
plt.title('loss')
plt.show()
#%% モデルパラメータ保存
model.save_weights('result/model/derepa004.h5')

#%% テスト
im = cv2.imread('test_data/renamed/0.jpg')

gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
faces = classifier.detectMultiScale(gray_im)
output_dir = 'result/fig/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if len(faces) == 0:
    print('cannot ditected!!')

for i, (x,y,w,h) in enumerate(faces):
    l = max([w,h])
    im_face = im[y:y+l, x:x+l]
    im_face = cv2.resize(im_face, (size,size))
    im_face = [im_face.astype('float32') / 255]
    r = model.predict(np.array(im_face), batch_size=2, verbose=1)
    res = r[0]
    for i, acc in enumerate(res):
        print(idol[i+1], '=', int(acc * 100))
    print('result---', idol[res.argmax()+1])
    output_path = os.path.join(output_dir,str(i)+'.'+f.split('.')[-1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    if res.argmax()+1 == 1:
        color=(0,0,255)
    else:
        color=(0,255,0)
    cv2.putText(im, idol[res.argmax()+1], (x, y), font,1, color=color)
    print('result---', idol[res.argmax()+1])
    cv2.rectangle(im, (x,y), (x+w,y+h), color=color, thickness=3)
cv2.imwrite(output_path, im)
plt.imshow(im)
plt.show()
