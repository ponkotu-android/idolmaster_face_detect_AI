# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 20:30:39 2018

@author: ponkotu_androido
"""

import os
import cv2
import glob
import numpy as np
import matplotlib.pylab as plt

#%%'tada_riina'
idol = {1:'honda_mio',
        2:'tada_riina'
        }
#%%
size = 100
ex = 35
# 特徴量ファイルをもとに分類器を作成
classifier = cv2.CascadeClassifier('etc/lbpcascade_animeface.xml')
faces = []

# 顔の検出
for i in idol.keys():
    print(idol[i])
    name = idol[i]
    for f in glob.glob('raw_data/'+name+'/*'):
        image = cv2.imread(f)
        #plt.imshow(image)
        #plt.show()
        print(f)
        # グレースケールで処理を高速化
        gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(image)
        output_dir = 're_data/faces3/'+name
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for i, (x,y,w,h) in enumerate(faces):
            l = max([w,h])
            # 一人ずつ顔を切り抜く
            try:
                face_image = image[y-ex:y+l+ex, x-ex:x+l+ex]
                face_image = cv2.resize(face_image,(size,size))
                output_path = os.path.join(output_dir,str(i)+str(ex)+f.split('\\')[-1])
                plt.imshow(face_image)
                plt.show()
                cv2.imwrite(output_path,face_image)
            except:
                pass
