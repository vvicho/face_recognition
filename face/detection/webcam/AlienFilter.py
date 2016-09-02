'''
Created on Aug 22, 2016

@author: daniel
'''

import cv2
import sys
import numpy as np
from tkinter import *
from utils import Point
from cmath import sqrt, atan
from math import pi, atan2

haarcascades_path = '/home/daniel/opencv/data/haarcascades/'
FACES_PATH = '/home/daniel/Downloads/Images/lfw-deepfunneled/'

# cascPath = haarcascades_path
faceCascade = cv2.CascadeClassifier(haarcascades_path + 'haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier(haarcascades_path + 'haarcascade_eye.xml')
right_eye_cascade = cv2.CascadeClassifier(haarcascades_path + 'haarcascade_lefteye_2splits.xml')
left_eye_cascade =  cv2.CascadeClassifier(haarcascades_path + 'haarcascade_righteye_2splits.xml')
profile_face_cascade = cv2.CascadeClassifier(haarcascades_path + 'haarcascade_profileface.xml')
nose_cascade = cv2.CascadeClassifier(haarcascades_path + 'Nariz.xml')
# smile_cascade =  cv2.CascadeClassifier(haarcascades_path + 'haarcascade_smile.xml')

video_capture = cv2.VideoCapture(0)

#Camera size
DETECTION_WIDTH = 640/1.5
#Shrink the image to run faster

frame_width, frame_height = video_capture.get(3), video_capture.get(4) # get width
scale = frame_width/DETECTION_WIDTH



video_capture.set(3, frame_width / scale)
video_capture.set(4, frame_height / scale)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equilizedImg = cv2.equalizeHist(gray)
    faces = faceCascade.detectMultiScale(
        equilizedImg,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(20, 20),
        flags=cv2.CASCADE_SCALE_IMAGE
#         flags=cv2.CASCADE_FIND_BIGGEST_OBJECT
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.circle(frame, ((x+w//2),(y+h//2)), (y+h)//2,  (255,0,0), 1)
#         cv2.ellipse(frame, ((x+w//2),(y+h//2)), (75 ,100), 0, 0, 360, 255, 1)

        roi_gray = equilizedImg[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(30,30))
        right_eyes = right_eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(30,30))
        left_eyes = left_eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(30,30))
        noses = nose_cascade.detectMultiScale(roi_gray)
        
#         if len(left_eyes) > 1 :
        for(ex, ey, ew, eh) in left_eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255,0,0),2)
#         elif len(right_eyes) > 1 :
#             for(ex, ey, ew, eh) in right_eyes:
#                 cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,255),2)
        
    # Display the resulting frame
    cv2.imshow('Video', frame)
#     canvas.create_image(0,0,frame)
    if(cv2.waitKey(10) == 27):
        break

    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()






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