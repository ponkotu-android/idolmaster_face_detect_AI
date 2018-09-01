# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 00:58:59 2018

@author: ponkotu_androido
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import cv2
import glob
import os
from PIL import Image, ImageDraw, ImageFont

from keras.models import Sequential  
from keras.layers.core import Dense, Dropout
from keras.optimizers import RMSprop

from make_model import mk_model
ex = 35
size = 100
idol = {1:'honda_mio',
        2:'tada_riina'}
inp_size = size
out_size = 2

model = mk_model(inp_size, out_size)
model.load_weights('result/model/derepa004.h5')
#%%
classifier = cv2.CascadeClassifier('etc/lbpcascade_animeface.xml')

faces = []

# 顔の検出
f='test_data/renamed/102.jpg'
im = cv2.imread(f)
print(f)

# グレースケールで処理を高速化
gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
faces = classifier.detectMultiScale(gray_im)
output_dir = 'result/fig/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if len(faces) == 0:
    print('cannot ditected!!')

for i, (x,y,w,h) in enumerate(faces):
    l = max([w,h])
    # 一人ずつ顔を切り抜く
    im_face = im[y-ex:y+l+ex, x-ex:x+l+ex]
    im_face = cv2.resize(im_face, (size,size))
    im_face = [im_face.astype('float32') / 255]
    r = model.predict(np.array(im_face), batch_size=5, verbose=1)
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

#%%


for i, acc in enumerate(res):
    print(idol[i+1], '=', int(acc * 100))
print('result---', idol[res.argmax()+1])