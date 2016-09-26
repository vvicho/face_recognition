'''
Created on Sep 21, 2016

@author: daniel
'''

import cv2
import sys
import time
import numpy as np
from face.detection.PreprocessImage import getPreprocessedFace
from face.recognition.Recognition import getSimilarity, learnCollectedFaces, reconstructFace

#Enums to define states
class TYPE:
    VIDEO, PICTURE = range(2)

class MODE:
    STARTUP, DETECTION, COLLECT_FACES, TRAINING, RECOGNITION, DELETE_ALL, END = range(7)

# Cascade file locations
haarCascadesPath = '/home/daniel/opencv/data/haarcascades/'
faceCascadeFile = haarCascadesPath + 'haarcascade_frontalface_alt2.xml'
eyeCascade1File = haarCascadesPath + 'haarcascade_lefteye_2splits.xml'
eyeCascade2File = haarCascadesPath + 'haarcascade_righteye_2splits.xml'

# Recognition Data
facerecAlgorithm = "FaceRecognizer.Fisherfaces"

# Desired face dimensions. getPreprocessedFace() will return a square face
faceWidth = 70
faceHeight = 70

# Parameters controlling how often to keep new faces when collecting them. 
# Otherwise the training set could look similar to each other
CHANGE_IN_IMAGE_FOR_COLLECTION = 0.3
CHANGE_IN_SECONDS_FOR_COLLECTION = 1.0

BORDER = 8 # Border between the GUI elements to the edge of the image.

# stacks to store the person face and name
preprocessedFaces = []
faceLabels = []

# Preprocess left & right sides of the face separately in case there is stronger light in one side.
preprocessLeftAndRightSeparately = True

# Sets how confident the face verification algorithm should be to decide if it is an unknown or known person
# A value roughly around seems OK for Eigenfaces or 0.7 for Fisherfaces, but you may want to adjust it for your
# conditions, and if you use a different Face Recognition algorithm
# Note that a higher threshold value means accepting more faces as known people,
# whereas lower values mean more faces will be classified as unknown.
UNKNOWN_PERSON_THRESHOLD = 0.7 

mMode = MODE.DETECTION
mNumPersons = 0
mLatestFaces = []
mDebug = False

runType = TYPE.VIDEO



def run():
    faceCascade, eyeCascade1, eyeCascade2 = initDetectors()
    
    recognizeAndTrain(None, faceCascade, eyeCascade1, eyeCascade2)
    return None
    
    
'''
    Load the face and 1 or 2 eye detection XML classifiers
'''
def initDetectors():
    faceCascade = eyeCascade1 = eyeCascade2 = None
    # Initialize face cascade
    try:
        faceCascade = cv2.CascadeClassifier(faceCascadeFile)
    except Exception as e:
        print "Error loading face cascade: " + e.__str__
        print "cascade name = " + faceCascadeFile
        sys.exit()
    
    # Initialize eye cascade 1
    try:
        eyeCascade1 = cv2.CascadeClassifier(eyeCascade1File)
    except Exception as e:
        print "Error loading eye cascade1 : " + e.__str__
        print "cascade name = " + eyeCascade1File
        sys.exit()
        
    # Initialize eye cascade 2
    try:
        eyeCascade2 = cv2.CascadeClassifier(eyeCascade2File)
    except Exception as e:
        if eyeCascade2File == '' or eyeCascade2File == None:
            print "There is no second Eye Cascade. Will continue with only one"
        else:
            print "Second eye cascade was not found. Will work with eye cascade 1 only"
            print "" + eyeCascade2File
        
        eyeCascade2 = None
        
        
    return faceCascade, eyeCascade1, eyeCascade2

def initWebcam():
    try:
        videoCapture = cv2.VideoCapture(0)
    except Exception as e:
        print "Could not open video camera"
        print e.__str__
        sys.exit()

def doStuff(src, faceCascade, eyeCascade1, eyeCascade2, oldPreprocessedFace):
    # Run the face recognition system on the src image. 
    # It will draw some things onto the given image, so make sure it is not read-only memory!
    global mMode
    identity = -1
    mTime = time.time()
    # Find face and preprocess it to have a standard size, contrast and brightness
    preprocessedFace, faceRect, leftEye, rightEye, searchedLeftEye, searchedRightEye = getPreprocessedFace(src, faceWidth, faceCascade, eyeCascade1, eyeCascade2, preprocessLeftAndRightSeparately)
    print "0000000000000000000000000000000000000000000000000000"
    print preprocessedFace
    print "0000000000000000000000000000000000000000000000000000"
    print faceRect
    print "0000000000000000000000000000000000000000000000000000"
    print leftEye
    print "0000000000000000000000000000000000000000000000000000"
    print rightEye
    print "0000000000000000000000000000000000000000000000000000"
    print searchedLeftEye
    print "0000000000000000000000000000000000000000000000000000"
    print searchedRightEye
    print "1111111111111111111111111111111111111111111111111111"
    
    gotFaceAndEyes = False
    if preprocessedFace is not None:
        gotFaceAndEyes = True
        
    # Draw an anti-aliased rectangle around the detected face
    if faceRect is not None and len(faceRect) > 0:
        print faceRect
        
        cv2.rectangle(src, (faceRect[0], faceRect[1]), (faceRect[2], faceRect[3]), (255,255,0), 2, cv2.cv.CV_AA) # Check faceRect data
        
        eyeColor = cv2.cv.CV_RGB(0,255,255)
        
        print "Check circle function for python"
        if leftEye[0] >= 0:
            print leftEye
            leftEyeCenterX = cv2.cv.Round(faceRect[0]+leftEye[0])
            leftEyeCenterY = cv2.cv.Round(faceRect[1]+leftEye[1])
#             leftEyeCenterX = cv2.cv.Round((leftEye[0] + leftEye[2])/2.0)
#             leftEyeCenterY = cv2.cv.Round((leftEye[1] + leftEye[3])/2.0)
            radius = 6
            cv2.circle(src, (leftEyeCenterX, leftEyeCenterY), radius, (200,200,0)) # Check circle for python
        if rightEye[0] >= 0:
            rightEyeCenterX = cv2.cv.Round(faceRect[0]+rightEye[0])
            rightEyeCenterY = cv2.cv.Round(faceRect[1]+rightEye[1])
#             rightEyeCenterX = cv2.cv.Round((rightEye[0] + rightEye[2])/2.0)
#             rightEyeCenterY = cv2.cv.Round((rightEye[1] + rightEye[3])/2.0)
            radius = 6
            cv2.circle(src, (rightEyeCenterX, rightEyeCenterY), radius, (200,200,0)) # Check circle for python
            
        if mMode == MODE.DETECTION:
            # Don't do anything special
            pass
        elif mMode == MODE.COLLECT_FACES:
            # Check if we have detected face
            if gotFaceAndEyes:
                # Check if this face looks somewhat different from the previously collected face
                imageDiff = 10000000000.0
                if oldPreprocessedFace:
                    imageDiff = getSimilarity(preprocessedFace, oldPreprocessedFace)
                
                # Also record when it happened 
                currentTime = time.time()
                timeDiff = currentTime - mTime
                
                #Only process the face if it is noticeably different from the previous frame and there has been a noticeable time gap
                if (imageDiff > CHANGE_IN_IMAGE_FOR_COLLECTION and timeDiff > CHANGE_IN_SECONDS_FOR_COLLECTION):
                    # Also add the mirror image to the training set, so we hace more training data, as well as to deal with faces looking to the left or to the right
                    mirroredFace = cv2.flip(preprocessedFace, 1)
                    
                    preprocessedFaces.append(preprocessedFace)
                    preprocessedFaces.append(mirroredFace)
                    faceLabels.append("")
                    faceLabels.append("")
                    
                    # Keep a reference to the latest face of each person
                    '''
                    mLatestFaces[mSelectedPerson] = len(preprocessedFaces) - 2 # Point to the non-mirrored face
                    '''
                    # Show the nomber of collected faces. But since we are also storing mirrored faces, 
                    # Just show the user how many think he has stored
                    
                    print "Saved face {0} for person {1} ".format(len(preprocessedFaces) / 2, "Daniel")
                    
                    # Make a white flash on the face, so the user knows a photo has been taken
                    displayedFaceRegion = src[faceRect[1]:faceRect[1] + faceRect[3], faceRect[0]:faceRect[0] + faceRect[2]]
                    displayedFaceRegion += cv2.cv.CV_RGB(90,90,90)
                    
                    # Keep a copy of the processed face, to compare on next iteration
                    mTime = time.time()
                    return preprocessedFace
                    
                    
        elif mMode == MODE.TRAINING:
            
            # Check if there is enough data to train from. For Eigenfaces, we can learn just one person if we want, but for FisherFaces,
            # we need at least 2 people otherwise it will crash
            haveEnoughData = True
            if facerecAlgorithm == "FaceRecognizer.Fisherfaces":
                if mNumPersons < 2 or mNumPersons == 2 and mLatestFaces[1] < 0:
                    print "Warning: Fisherfaces needs at least 2 people, otherwise there is nothing to differentiate! Collect more data."
                    haveEnoughData = False
            
            if mNumPersons < 1 or len(preprocessedFaces) <= 0 or len(preprocessedFaces) != len(faceLabels):
                print "Warning: Need some training data before it can be learnt! Collect more data."
                haveEnoughData = False
                
            if haveEnoughData:
                model = learnCollectedFaces(preprocessedFaces, faceLabels, facerecAlgorithm)
                
                if mDebug:
#                     showTrainingDebugData(model, faceWidth, faceHeight)
                    pass
                
                mMode = MODE.RECOGNITION
            else:
                mMode = MODE.COLLECT_FACES
        elif mMode == MODE.RECOGNITION:
            if gotFaceAndEyes and len(preprocessedFaces) > 0 and len(preprocessedFaces) == len(faceLabels):
                reconstructedFace = reconstructFace(model, preprocessedFace)
                if mDebug:
                    if len(reconstructedFace) > 0:
                        cv2.imshow("reconstructedFace", reconstructedFace)
                        
                # Verify whether the reconstructed face looks like the preprocessed face, otherwise it is probably an unknown person
                similarity = getSimilarity(preprocessedFace, reconstructedFace)
                
                
                if(similarity < UNKNOWN_PERSON_THRESHOLD):
                    # Identify who the person is in the preprocessed face image.
                    identity = model.predict(preprocessedFace)
                    outStr = str(identity) 
                else:
                    # Since the confidence is low, assume it is an unknown person
                    outStr = "Unknown"
                
                print "Identity: {0]. Similarity: {1}".format(outStr, similarity)
                
                #Show the confidence rating for the recognition in the mid-top of the display
                cx = (len(src[0]) - faceWidth) / 2
                ptBottomRight = (cx - 5, BORDER + faceHeight)
                ptTopLeft = (cx - 15, BORDER)
                # Draw a gray line showing the threshold for an "unknown" person
                ptThreshold = (ptTopLeft[0], ptBottomRight[1] - (1.0 - UNKNOWN_PERSON_THRESHOLD) * faceHeight)
                cv2.rectangle(src, ptThreshold, (ptBottomRight[0], ptThreshold[1]), cv2.cv.CV_RGB(200,200,200), 1, cv2.CV_AA)
                # Crop the confidence rating between 0.0 to 1.0, to show in the bar.
                confidenceRatio =  1.0 - min(max(similarity, 0.0), 1.0)
                ptConfdence = (ptTopLeft[0], ptBottomRight[1] - confidenceRatio * faceHeight)
                # Show the light-blue confidence bar
                cv2.rectangle(src, ptConfdence, ptBottomRight, cv2.cv.CV_RGB(0,255,255), cv2.cv.CV_FILLED, cv2.CV_AA)
                # Show the gray border of the bar
                cv2.rectangle(src, ptTopLeft, ptBottomRight, cv2.cv.CV_RGB(200,200,200), 1, cv2.CV_AA)
                
        elif mMode == MODE.DELETE_ALL:
            mSelectedPerson = -1
            mNumPersons = 0
            mLatestFaces = []
            preprocessedFaces = []
            faceLabels = []
            oldPreprocessedFace = np.array()
        else:
            print "ERROR: Invalid run mode {}".format(mMode)
            sys.exit()
            
    return None

def recognizeAndTrain(src, faceCascade, eyeCascade1, eyeCascade2):
    
    oldPreprocessedFace = None
    oldTime = 0
    # Start in detection mode
    mMode = MODE.DETECTION
    cam = cv2.VideoCapture(0)
    
    if runType == TYPE.PICTURE:
        # Run once for pictures
        doStuff(src, faceCascade, eyeCascade1, eyeCascade2)
    else:
        # Run forever until user hits esc in case it is video 
        while True:
            ret, frame = cam.read()
            oldPreprocessedFace = doStuff(frame, faceCascade, eyeCascade1, eyeCascade2, oldPreprocessedFace)
            cv2.imshow('Video', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cam.release()
                cv2.destroyAllWindows()
                break
            
        
run()

# cam = cv2.VideoCapture(0)
# while True:
#     ret, frame = cam.read()
#     cv2.imshow('bla', frame)
#     
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cam.release()
# cv2.destroyAllWindows()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
    
    