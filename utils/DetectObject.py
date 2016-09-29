'''
Created on Sep 6, 2016

@author: daniel
'''

import cv2
import numpy as np
import utils.Utilities as uti
import utils
import time
from math import atan2, pi

drawFaceRectangles = True

mDebug = False

def myPrint(obj, flag=False):
    global mDebug
    if mDebug or flag:
        print obj 


def drawFaceOutline(img, x,y,w,h, scale):
    
            x = cv2.cv.Round(x * scale)
            y = cv2.cv.Round(y * scale)
            w = cv2.cv.Round(w * scale)
            h = cv2.cv.Round(h * scale)
            myPrint((x, y, w, h)) 
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
    

'''
    Search for objects such as faces or eyes in the image using the given parameters.
    Can use haar cascades or LBP cascades for face detection, eye, mouth
    Input is temporarily shrunk to 'scaleWidth' for faster detection, since 200 is enough to find faces
    Returns
'''
def detectObjectsCustom(img, cascade, scaleWidth, flags, minFeatureSize, searchScaleFactor, minNeighbors, firstDetection, details):
    if img is None:
        print "img is None"
        return None
    
    # Will transform image to grayscale if it is the first run 
    if firstDetection:
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
    else:
        gray = img
    
#     print len(img[0])
#     print scaleWidth
    scale = len(img[0]) / float(scaleWidth)
    
    if len(img) > scaleWidth:
        # Shrink image to run faster
        scaleHeight = cv2.cv.Round(len(img) / scale )
        myPrint((scaleWidth, scaleHeight), False )
        inputImg = cv2.resize(gray, (scaleWidth, scaleHeight))
    else:
        # Access input image since it is already small
        inputImg = gray
    
    # Standarize the brightness and contrast to imporve dark images
    equalizedImage = cv2.equalizeHist(inputImg)
    
    #Detect objects in the small greyscale equilized image
    objects = cascade.detectMultiScale(equalizedImage,
        scaleFactor=searchScaleFactor,
        minNeighbors=minNeighbors,
        minSize=minFeatureSize,
        # equalizedImage, searchScaleFactor, minNeighbors, flags, minFeatureSize
                                       )
    myPrint('Cascade')
    myPrint(type(objects))
    myPrint(objects)
    
    if drawFaceRectangles:
        for (x,y,w,h) in objects:
            drawFaceOutline(img, x, y, w, h, scale)
        
    
    #Enlarge the results if the image was shrunk before
    if len(objects) > 0:
        for i in range(len(objects)):
            objects[i] = [objects[i][0] * scale, objects[i][1] * scale, objects[i][2] * scale, objects[i][3] * scale]
    myPrint('multiply')
    myPrint(objects)
    
    
    # Make sure the object is completely within the image, in case it was on a border
    for i in range(len(objects)):
        if objects[i][0] < 0:
            objects[i][0] = 0
        if objects[i][1] < 0:
            objects[i][1] = 0
        if objects[i][0] + objects[i][2] > len(img[i]):
            objects[i][0] = len(img[i]) - objects[i][2]
        if objects[i][1] + objects[i][3] > len(img):
            objects[i][1] = len(img) - objects[i][3]

    myPrint('adjusted')
    myPrint(objects)
    
    return objects        


'''
Search for just a single object in the image.
Input is shrunk for faster detection
Returns singe largest object
'''
def detectLargestObject(img, cascade, scaleWidth, details):
    #Only search 1 object
    flags = cv2.CASCADE_FIND_BIGGEST_OBJECT
    
    #smallest object size
    minFeatureSize = (30,30)
    #How detailed should the search be
    searchScaleFactor = 1.15
    #How much the dections should be filtered out
    minNeighbors = 3
    
    #Perform detection
    objects = detectObjectsCustom(img, cascade, scaleWidth, flags, minFeatureSize, searchScaleFactor, minNeighbors, True, "")
    if (objects is not None) and len(objects) >= 1:
        largestObject = objects[0]
    else:
        largestObject = None
        print("No object found in " + details)
    
    return largestObject


'''
Search for many objects in the image
Return recognized objects
'''
def detectManyObjects(img, cascade, scaleWidth, firstDetection, details):
    flags = cv2.CASCADE_SCALE_IMAGE
    
    minFeatureSize = (30,30)
    
    searchScaleFactor = 1.15
    
    minNeighbors = 3
    
    return detectObjectsCustom(img, cascade, scaleWidth, flags, minFeatureSize, searchScaleFactor, minNeighbors, firstDetection, details)    

# Rotate the face so that the two eyes are horizontal
# Scale the face so that the distance between the two eyes is always the same
# Translate the face so that the eyes are always centered horizontally and at the desired height
# Crop the outer parts of the face, since we want to crop away the image background, hair, forehead, ears and chin

