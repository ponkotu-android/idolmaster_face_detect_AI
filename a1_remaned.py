# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 00:29:31 2018

@author: ponkotu_androido
"""
import shutil, glob
i=1

for f in glob.glob('test_data/raw/*'):
    shutil.copy(f, 'test_data/renamed/'+str(i)+'.'+f.split('.')[-1])
    i = i+1
