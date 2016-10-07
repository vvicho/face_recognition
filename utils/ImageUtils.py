'''
Created on Oct 6, 2016

@author: daniel
'''

import cv2
from utils.Utilities import resize_img
import numpy as np

maxImgSize = 200

def createPicLayout(photo1, photo2):
    
    resize_img(photo1, 200)
    resize_img(photo2, 200)
    
    biggestPhoto = None
    if len(photo1) > len(photo2):
        biggestPhoto = photo1
    else:
        biggestPhoto = photo2
    
    mixedPhoto = np.zeros(np.zeros(0))
    for i in xrange(len(biggestPhoto)):
        mixedPhoto.append()
    
    
    return mixedPhoto

a = np.array([0,0])
print a
a = np.append(a, [1,2])
b = np.array([a])
b = np.append(b, np.array([2,3]))
# a = np.array(a, [1,2])
print a
print b