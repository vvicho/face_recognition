'''
Created on Jun 23, 2016

@author: daniel
'''

import cv2
import numpy as np
import utils.Utilities as uti
import utils
import time
from math import atan2, pi

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

draw_eyes = 0

face_cascade = cv2.CascadeClassifier(haarcascades_path + 'haarcascade_frontalface_alt2.xml')
eyeglasses_cascade = cv2.CascadeClassifier(haarcascades_path + 'haarcascade_eye_tree_eyeglasses.xml')
eye_cascade = cv2.CascadeClassifier(haarcascades_path + 'haarcascade_eye.xml')
right_eye_cascade = cv2.CascadeClassifier(haarcascades_path + 'haarcascade_lefteye_2splits.xml')
left_eye_cascade =  cv2.CascadeClassifier(haarcascades_path + 'haarcascade_righteye_2splits.xml')
smile_cascade =  cv2.CascadeClassifier(haarcascades_path + 'haarcascade_smile.xml')
fullbody_cascade = cv2.CascadeClassifier(haarcascades_path + 'haarcascade_fullbody.xml')


def detect(filename=None, folder=None, num_of_people=people, num_of_pics=pic_per_person):
    path = FACE_LIBRARY + 'IMG_20160619_152455.jpg'
    img = cv2.imread(path)
    face_array = []

    # print(imgs)
    index=1
    
    if resize:
        res = uti.resize_img(img, 480)
    else:
        res = img
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    equalizedImg = cv2.equalizeHist(gray)
         
    faces = face_cascade.detectMultiScale(equalizedImg, scale_factor, num_neighbors, cv2.CASCADE_FIND_BIGGEST_OBJECT)
#         print(photo.photo_name)
    if print_photos:
        for(x,y,w,h) in faces:
            #Paint face
            cv2.rectangle(res, (x,y), (x+w, y+h), (255,0,0),2)
            
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
        cv2.imshow('Photo', res)
        print('face fotos ' + str(len(face_array)))
        for n, photo in enumerate(face_array):
            print()
            cv2.imshow('photo' + str(n), photo)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
#     print(type(["hola"]), type("hola"))
    
    

# detect(folder="George_W_Bush")
# detect(filename='George_W_Bush_0011.jpg')
detect()

# Rotate the face so that the two eyes are horizontal
# Scale the face so that the distance between the two eyes is always the same
# Translate the face so that the eyes are always centered horizontally and at the desired height
# Crop the outer parts of the face, since we want to crop away the image background, hair, forehead, ears and chin

def warpAffine(left_eye, right_eye, gray):
    
    #Get the center between the two eyes
    eyes_center_x = (left_eye.x + right_eye.x) / 2
    eyes_center_y = (left_eye.y + right_eye.y) / 2
    eyes_center = (eyes_center_x, eyes_center_y)
    
    #Get the angle between the 2 eyes
    dy = right_eye.y - left_eye.y
    dx = right_eye.x - left_eye.x
    len = (dx**2 + dy**2) ** 0.5
    
    #Convert radians to degrees
    angle = atan2(dy, dx) * 180.0/ pi
    
    # Hand measurements shown that the left eye center should
    # ideally be roughly at (0.16, 0.14) of a scaled face image
    DESIRED_LEFT_EYE_X  = 0.16
    DESIRED_RIGHT_EYE_X = (1.0 - 0.16)
    
    # Get the amount we need to scale the image to be the desired 
    # fixed size we want
    
    DESIRED_FACE_WIDTH  = 70 
    DESIRED_FACE_HEIGHT = 70
    
    desired_len = (DESIRED_RIGHT_EYE_X - 0.16)
    scale = desired_len * DESIRED_FACE_WIDTH / len
    
    #Get the transformation matrix for the desired angle & Size
    rot_mat = cv2.getRotationMatrix2D(eyes_center, angle, scale)
    
    #shift the center of the eyes to be the desired center
    ex = DESIRED_FACE_WIDTH * 0.5 - eyes_center.x
    ''' !!!!!!!!!!!   '''
    ey = DESIRED_FACE_HEIGHT * DESIRED_LEFT_EYE_X - eyes_center.y 
    ''' !!!!!!!!!!!   '''
    
#     rot_mat.at(0,2) += ex
#     rot_mat.at(1,2) += ey 
    rot_mat[0,2] += ex
    rot_mat[1,2] += ey 
    
    # Transform the face image to the desired angle & size & position!
    # Also clear the transformed image background to a default grey
    warped = np.mat((DESIRED_FACE_HEIGHT, DESIRED_FACE_WIDTH), np.uint8)
    
    cv2.warpAffine(gray, warped, rot_mat, len(warped))
    
    