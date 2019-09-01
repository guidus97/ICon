 # -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 11:00:32 2019

@author: utente
"""

import numpy as np
import os
import cv2

def read_categories_dict(filename):
    d={}
    with open(filename) as f:
        for line in f:
            (key,val1,val2,val3) = line.split()
            d[int(key)]=[val1,val2,val3]
    return d


def load_and_resize(dirpath):

    x_test=[]

    for img in os.listdir(dirpath):
        image_file=cv2.imread(dirpath+'/'+img)
        
        if image_file is not None:
            resized=cv2.resize(image_file,(32,32))
            x_test.append(resized)
            
        else: print('Error')
    return np.array(x_test)


def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)