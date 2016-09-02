'''
Created on Sep 2, 2016

@author: daniel
'''

# import cv2
import cv2


def detectBothEyes(face, cascade_classifier, eye_cascade, left_eye, right_eye,
                   searched_left_eye, searched_right_eye):
    
    eye_sx = 0.16
    eye_sy = 0.26
    eye_sw = 0.30
    eye_sh = 0.28
    
    height, width = face.shape[:2]
    
print(round(02.365))

