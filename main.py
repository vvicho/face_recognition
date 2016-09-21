'''
Created on Sep 21, 2016

@author: daniel
'''

import cv2
import sys
from face.detection.PreprocessImage import getPreprocessedFace

#Enums to define states
class TYPE:
    VIDEO, PICTURE = range(2)

class MODE:
    MODE_STARTUP, MODE_DETECTION, MODE_COLLECT_FACES, MODE_TRAINING, MODE_RECOGNITION, MODE_DELETE_ALL, MODE_END = range(7)

# Cascade file locations
haarCascadesPath = '/home/daniel/opencv/data/haarcascades/'
faceCascadeFile = haarCascadesPath + 'haarcascade_frontalface_alt2.xml'
eyeCascade1File = haarCascadesPath + 'haarcascade_lefteye_2splits.xml'
eyeCascade2File = haarCascadesPath + 'haarcascade_righteye_2splits.xml'

# Desired face dimensions. getPreprocessedFace() will return a square face
faceWidth = 70
faceHeight = 70


# Preprocess left & right sides of the face separately in case there is stronger light in one side.
preprocessLeftAndRightSeparately = True

videoCapture = None

runType = TYPE.PICTURE



def run():
    faceCascade, eyeCascade1, eyeCascade2 = initDetectors()
    
    recognizeAndTrain()
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

def recognizeAndTrain(src, faceCascade, eyeCascade1, eyeCascade2):
    
    oldTime = 0
    # Start in detection mode
    mMode = MODE.MODE_DETECTION
    
    # Run forever until user hits esc in case it is video 
        
    while True:
        
        preprocessedFace = getPreprocessedFace(src, faceWidth, faceCascade, eyeCascade1, eyeCascade2, preprocessLeftAndRightSeparately)
        
        
        
        if runType == TYPE.PICTURE:
            break
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
    
    