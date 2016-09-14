'''
Created on Jun 23, 2016

@author: daniel
'''

import cv2
import numpy as np
import utils.Utilities as uti
import utils.DetectObject as det
import time
from math import atan2, pi
import caffe


start_time = time.time()

# Directory paths
haarcascades_path = '/home/daniel/opencv/data/haarcascades/'
FACE_LIBRARY = '/home/daniel/Downloads/Images/Daniel/test/'             
                 
# Number of photos configuration
people = -1
pic_per_person = -1 # -set to -1 to bring all photos from a person

# Face Detection coniguration
scale_factor = 1.05
num_neighbors = 3

# other configurations
print_photos = True
write_log = True
resize = True 

draw_eyes = 1

face_cascade = cv2.CascadeClassifier(haarcascades_path + 'haarcascade_frontalface_alt2.xml')
eyeglasses_cascade = cv2.CascadeClassifier(haarcascades_path + 'haarcascade_eye_tree_eyeglasses.xml')
eye_cascade = cv2.CascadeClassifier(haarcascades_path + 'haarcascade_eye.xml')
right_eye_cascade = cv2.CascadeClassifier(haarcascades_path + 'haarcascade_lefteye_2splits.xml')
left_eye_cascade =  cv2.CascadeClassifier(haarcascades_path + 'haarcascade_righteye_2splits.xml')
smile_cascade =  cv2.CascadeClassifier(haarcascades_path + 'haarcascade_smile.xml')
fullbody_cascade = cv2.CascadeClassifier(haarcascades_path + 'haarcascade_fullbody.xml')


def detect(filename=None, folder=None, num_of_people=people, num_of_pics=pic_per_person):
    path = FACE_LIBRARY + 'IMG_20160619_152455(dif).jpg'
    img = cv2.imread(path)
    face_array = []
    
    # print(imgs)
    index=1
    
    if resize:
        res = uti.resize_img(img, 600)
    else:
        res = img
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    print(len(gray), len(gray[0]))
    equalizedImg = cv2.equalizeHist(gray)
    
    print('img shape {}'.format(img.shape))
    print('gray shape {}'.format(gray.shape))
    print('equa  shape {}'.format(equalizedImg.shape))
   
    #Test
    out = det.detectObjectsCustom(img, face_cascade, 600, None, (20, 20), scale_factor, num_neighbors, True, 'Face')
    #End Test
    
    faces = face_cascade.detectMultiScale(equalizedImg, scale_factor, num_neighbors, cv2.CASCADE_FIND_BIGGEST_OBJECT)
    print('faces\n')
    print(faces) 
    if print_photos:
        for i in range(len(out)):
            print(i)
            #Paint face
            (x,y,w,h) = (out[i][0], out[i][1], out[i][2], out[i][3])
#             face = (x,y,w,h)
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),2)
            
            #Get all facial data
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = res[y:y+h, x:x+w]
            face_array += [roi_color]
            left_eyes = left_eye_cascade.detectMultiScale(roi_gray)
            right_eyes = right_eye_cascade.detectMultiScale(roi_gray)
            eyes = eye_cascade.detectMultiScale(roi_gray)
#             smiles = smile_cascade.detectMultiScale(roi_gray)
#             bodies = fullbody_cascade.detectMultiScale(roi_gray)
            
            #Paint both eyes
            if draw_eyes == 0 :
                for(ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,255),2)
            #Paint left eyes
            elif draw_eyes == 1:
                for(ex, ey, ew, eh) in left_eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0),2)
            #Paint right eyes
            else:
                for(ex, ey, ew, eh) in right_eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,0,255),2)
            
            print("Eyes : {0}".format(len(eyes)))
            print("Left eyes : {0}".format(len(left_eyes)))
            print("Right eyes : {0}".format(len(right_eyes)))
    #     print("finished")
            
        total_time = time.time() - start_time  
        print("----- %s seconds ----" %total_time)
        cv2.imshow('Photo', img)
        print('face fotos ' + str(len(face_array)))
#         for n, photo in enumerate(face_array):
#             print()
#             cv2.imshow('photo' + str(n), photo)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
#     print(type(["hola"]), type("hola"))
    
    

# detect(folder="George_W_Bush")
# detect(filename='George_W_Bush_0011.jpg')
detect()



