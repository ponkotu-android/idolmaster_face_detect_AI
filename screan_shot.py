# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 14:40:48 2018

@author: ponkotu_androido
"""

from PIL import ImageGrab
from time import sleep
import os

d = 'dereste00mio/'
output_dir = 'SS/'+d
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
l=120
# full screen
for t in range(l):
    ImageGrab.grab().save(output_dir+str(t)+'.jpg')
    sleep(1)
    print(t)