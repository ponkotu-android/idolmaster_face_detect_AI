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
#%%
#左右反転、閾値処理、ぼかしで８倍の水増し
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
import os

def scratch_image(img, flip=True, thr=True, filt=True):
    # 水増しの手法を配列にまとめる
    methods = [flip, thr, filt]
    # ぼかしに使うフィルターの作成
    filter1 = np.ones((3, 3))
    # オリジナルの画像データを配列に格納
    images = [img]
    # 手法に用いる関数
    scratch = np.array([
        lambda x: cv2.flip(x, 1),
        lambda x: cv2.threshold(x, 100, 255, cv2.THRESH_TOZERO)[1],
        lambda x: cv2.GaussianBlur(x, (5, 5), 0),
    ])
    # 加工した画像を元と合わせて水増し
    doubling_images = lambda f, imag: np.r_[imag, [f(i) for i in imag]]

    for func in scratch[methods]:
        images = doubling_images(func, images)
    return images
    
# 画像の読み込み

for i in idol.keys():
    print(idol[i])
    name = idol[i]
    in_dir = 're_data/faces3/'+name+'/*'
    in_jpg=glob.glob(in_dir)
    img_file_name_list=os.listdir('re_data/faces/'+name+'/')
    for i in range(len(in_jpg)):
        print(str(in_jpg[i]))
        img = cv2.imread(str(in_jpg[i]))
        scratch_face_images = scratch_image(img)
        for num, im in enumerate(scratch_face_images):
            fn, ext = os.path.splitext(img_file_name_list[i])
            file_name=os.path.join('re_data/faces3/'+name+'/kakou_'+str(i)+str(num)+'.jpg')
            cv2.imwrite(str(file_name) ,im)
            
        
        
        
        
        
        
        