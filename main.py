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
import utils.CSVutils as csv
from utils.CSVutils import getNameTranslation, updateCSVFile
import Tkinter
from gui import PopupWindow
from multiprocessing.pool import ThreadPool
from utils.Utilities import faceCenterChanged, createNewFolder,\
    getNextLabelNumber




# nameArr = [(0, "Other"), (1, "Daniel"), (2, "Felicia"), (3,"Haruka"), (4, "Pao"), (5, "Rubi")]

#Enums to define states
class TYPE:
    VIDEO, PICTURE = range(2)

class MODE:
    STARTUP, DETECTION, COLLECT_FACES, TRAINING, RECOGNITION, TEST, DELETE_ALL, END = range(8)

runType = TYPE.VIDEO

imgFolderPath = "/home/daniel/workspace/Project/Images/Pics/Training/"
testFolderPath = "/home/daniel/workspace/Project/Images/Pics/Test/"
# imgFolderPath = "/home/daniel/workspace/Project/Images/yalefaces/jpeg/Training/"
# imgFolderPath = "/home/daniel/workspace/Project/Images/lfw/"
# testFolderPath = "/home/daniel/workspace/Project/Images/lfw/George_W_Bush/"
# testFolderPath = "/home/daniel/workspace/Project/Images/yalefaces/jpeg/Test/"

# Cascade file locations
haarCascadesPath = '/home/daniel/opencv/data/haarcascades/'
faceCascadeFile = haarCascadesPath + 'haarcascade_frontalface_alt2.xml'
eyeCascade1File = haarCascadesPath + 'haarcascade_lefteye_2splits.xml'
eyeCascade2File = haarCascadesPath + 'haarcascade_righteye_2splits.xml'

# Recognition Data
facerecAlgorithm = "Eigenfaces"

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
# currentProcessedFaces = []
# currentRecognized = []
faceLabels = []
nameTranslations = {}
# testPreProcessedFaces = []
# testFaceLables = []

# Preprocess left & right sides of the face separately in case there is stronger light in one side.
preprocessLeftAndRightSeparately = True

# Sets how confident the face verification algorithm should be to decide if it is an unknown or known person
# A value roughly around seems OK for Eigenfaces or 0.7 for Fisherfaces, but you may want to adjust it for your
# conditions, and if you use a different Face Recognition algorithm
# Note that a higher threshold value means accepting more faces as known people,
# whereas lower values mean more faces will be classified as unknown.
UNKNOWN_PERSON_THRESHOLD = 0.7 

mMode = MODE.COLLECT_FACES
mNumPersons = 0
mLatestFaces = []
mDebug = False
mShowImg = True
mSelectedPerson = -1
mStoreCollectedFaces = False
model = None
modelPath = "/home/daniel/workspace/Project/modelData.xml"
mTrainingTime = -1
mReadFromFiles = False

# Paint options
mFaceFrameColor = (0,255,0)
mEyeCircleColor = (200,200,0)
mPaintFaceFrame = True
mPaintEyeCircle = True

# previous faceRect to calculate centers and changes in position
prevFaceRect = None
prevLeftEye = None
prevRightEye = None

# Screen and frame details
root = Tkinter.Tk()
screenWidth = root.winfo_screenwidth()
screenHeight = root.winfo_screenheight()
camFrame = None
 


def myPrint(obj, flag=False):
    global mDebug
    if mDebug or flag:
        print obj 

def init():
    global imgFolderPath, nameTranslations
    nameTranslations = getNameTranslation(imgFolderPath)
    updateCSVFile(imgFolderPath)
    
    
def run():
    init()
    faceCascade, eyeCascade1, eyeCascade2 = initDetectors()
    print nameTranslations
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
    global mMode, mDebug, preprocessedFaces, faceLabels, model, mNumPersons, mLatestFaces, mSelectedPerson, mTrainingTime, mFaceFrameColor, mPaintEyeCircle, mPaintFaceFrame, mEyeCircleColor, prevFaceRect, prevLeftEye, prevRightEye, testPreProcessedFaces, mShowImg
    identity = -1
    mTime = time.time()
    # Find face and preprocess it to have a standard size, contrast and brightness
    preprocessedFace, faceRect, leftEye, rightEye, searchedLeftEye, searchedRightEye = getPreprocessedFace(src, faceWidth, faceCascade, eyeCascade1, eyeCascade2, preprocessLeftAndRightSeparately)
    currFaceRect = faceRect
    currLeftEye = leftEye
    currRightEye = rightEye
    myPrint ("000000000000perprocessedface0000000000000000000000000000000000000000")
    myPrint (preprocessedFace)
    myPrint ("000000000000facerect0000000000000000000000000000000000000000")
    myPrint (faceRect)
    myPrint ("0000000000000lefteye000000000000000000000000000000000000000")
    myPrint (leftEye)
    myPrint ("0000000000000righteye000000000000000000000000000000000000000")
    myPrint (rightEye)
    myPrint ("0000000000000searchedlefteye000000000000000000000000000000000000000")
    myPrint (searchedLeftEye)
    myPrint ("0000000000000searchedrighteye000000000000000000000000000000000000000")
    myPrint (searchedRightEye)
    myPrint ("1111111111111111111111111111111111111111111111111111")
        
    gotFaceAndEyes = False
    if preprocessedFace is not None:
        gotFaceAndEyes = True
        
    # Draw an anti-aliased rectangle around the detected face
    if faceRect is not None and len(faceRect) > 0:
        myPrint (faceRect)
        
        
        if not faceCenterChanged(camFrame, prevFaceRect, currFaceRect):
            currFaceRect = prevFaceRect
            currLeftEye = prevLeftEye
            currRightEye = prevRightEye
            
        if mPaintFaceFrame:
            cv2.rectangle(src, (currFaceRect[0], currFaceRect[1]), (currFaceRect[0] + currFaceRect[2], currFaceRect[1] + currFaceRect[3]), mFaceFrameColor, 2, cv2.cv.CV_AA) # Check faceRect data
        
        
        if mPaintEyeCircle:
            eyeColor = mEyeCircleColor
            
            if leftEye[0] >= 0:
                myPrint (leftEye)
                leftEyeCenterX = cv2.cv.Round(currFaceRect[0]+currLeftEye[0])
                leftEyeCenterY = cv2.cv.Round(currFaceRect[1]+currLeftEye[1] + 9)
    #             leftEyeCenterX = cv2.cv.Round((leftEye[0] + faceRect[2])/2.0)
    #             leftEyeCenterY = cv2.cv.Round((leftEye[1] + faceRect[3])/2.0)
                radius = 6
                cv2.circle(src, (leftEyeCenterX, leftEyeCenterY), radius, eyeColor) # Check circle for python
            if rightEye[0] >= 0:
                rightEyeCenterX = cv2.cv.Round(currFaceRect[0]+currRightEye[0])
                rightEyeCenterY = cv2.cv.Round(currFaceRect[1]+currRightEye[1] + 9)
    #             rightEyeCenterX = cv2.cv.Round((rightEye[0] + faceRect[2])/2.0)
    #             rightEyeCenterY = cv2.cv.Round((rightEye[1] + faceRect[3])/2.0)
                radius = 6
                cv2.circle(src, (rightEyeCenterX, rightEyeCenterY), radius, eyeColor) # Check circle for python
        
        prevFaceRect = currFaceRect
        prevLeftEye = currLeftEye
        prevRightEye = currRightEye
            
        if mMode == MODE.DETECTION:
            # Don't do anything special
            pass
        elif mMode == MODE.COLLECT_FACES:
            # Check if we have detected face
            if gotFaceAndEyes:
                # Check if this face looks somewhat different from the previously collected face
                imageDiff = 10000000000.0
#                 print oldPreprocessedFace
#                 print preprocessedFace
                if oldPreprocessedFace is not None:
                    imageDiff = getSimilarity(preprocessedFace, oldPreprocessedFace)
                
                myPrint("Image Diff = {}".format(imageDiff))
                # Also record when it happened 
                currentTime = time.time()
                timeDiff = currentTime - mTime
                
                #Only process the face if it is noticeably different from the previous frame and there has been a noticeable time gap
#                 print timeDiff
                if (imageDiff > CHANGE_IN_IMAGE_FOR_COLLECTION or timeDiff > CHANGE_IN_SECONDS_FOR_COLLECTION):
                    # Also add the mirror image to the training set, so we hace more training data, as well as to deal with faces looking to the left or to the right
                    print "Added new pic"
                    mirroredFace = cv2.flip(preprocessedFace, 1)
                    
                    preprocessedFaces.append(preprocessedFace)
                    preprocessedFaces.append(mirroredFace)
#                     newLabel = mSelectedPerson+getNextLabelNumber(nameTranslations)
#                     nameTranslations.update({newLabel: "Unknown"})
#                     faceLabels.append(mSelectedPerson + getNextLabelNumber(nameTranslations))
#                     faceLabels.append(mSelectedPerson + getNextLabelNumber(nameTranslations))
                    faceLabels.append(mSelectedPerson)
                    faceLabels.append(mSelectedPerson)
                    
                    
                    # Keep a reference to the latest face of each person
                    if mSelectedPerson >= 0 and len(mLatestFaces) > 0:
                        mLatestFaces[mSelectedPerson] = len(preprocessedFaces) - 2 # Point to the non-mirrored face
                    
                    # Show the nomber of collected faces. But since we are also storing mirrored faces, 
                    # Just show the user how many think he has stored
                    
                    print "Saved face {0} for person {1} ".format(len(preprocessedFaces) / 2, str(mSelectedPerson))
                    
                    # Make a white flash on the face, so the user knows a photo has been taken
                    displayedFaceRegion = src[currFaceRect[1]:currFaceRect[1] + currFaceRect[3], currFaceRect[0]:currFaceRect[0] + currFaceRect[2]]
                    displayedFaceRegion[np.where(True)] = [0,255,255]
#                     displayedFaceRegion += cv2.cv.CV_RGB(90,90,90)
                    
                    # Keep a copy of the processed face, to compare on next iteration
                    mTime = time.time()
                    
                return preprocessedFace
                    
                    
        elif mMode == MODE.TRAINING:
            
            # Check if there is enough data to train from. For Eigenfaces, we can learn just one person if we want, but for FisherFaces,
            # we need at least 2 people otherwise it will crash
            haveEnoughData = True
            if facerecAlgorithm == "Fisherfaces":
                if mNumPersons < 2 or mNumPersons == 2 and mLatestFaces[1] < 0:
                    print "Warning: Fisherfaces needs at least 2 people, otherwise there is nothing to differentiate! Collect more data."
                    haveEnoughData = False 
            
            if mNumPersons < 1 or len(preprocessedFaces) <= 0 or len(preprocessedFaces) != len(faceLabels):
                print "Warning: Need some training data before it can be learnt! Collect more data."
                haveEnoughData = False
            
            if haveEnoughData:
                tempTime = time.time()
                model = learnCollectedFaces(preprocessedFaces, faceLabels, facerecAlgorithm, model)
                mTrainingTime = time.time() - tempTime
                print mTrainingTime
                
                if mDebug:
#                     showTrainingDebugData(model, faceWidth, faceHeight)
                    pass
                
                mMode = MODE.RECOGNITION
            else:
                mMode = MODE.COLLECT_FACES
        elif mMode == MODE.RECOGNITION:
#             if gotFaceAndEyes and len(preprocessedFaces) > 0 and len(preprocessedFaces) == len(faceLabels):
            if model is not None:
                reconstructedFace = reconstructFace(model, preprocessedFace)
                if mShowImg :
                    if len(reconstructedFace) > 0:
                        cv2.imshow("reconstructedFace", reconstructedFace)
                        
                # Verify whether the reconstructed face looks like the preprocessed face, otherwise it is probably an unknown person
                similarity = getSimilarity(preprocessedFace, reconstructedFace)
                
                
                if(similarity < UNKNOWN_PERSON_THRESHOLD):
                    # Identify who the person is in the preprocessed face image.
                    identity = model.predict(preprocessedFace)
                    outStr = getPersonName(identity[0])  
                else:
                    # Since the confidence is low, assume it is an unknown person
                    outStr = "Unknown"
                
                myPrint("Identity: {0}. Similarity: {1}".format(outStr, similarity))
                
                #Show the confidence rating for the recognition in the mid-top of the display
                cx = (len(src[0]) - faceWidth) / 2
                ptBottomRight = (cx - 5, BORDER + faceHeight)
                ptTopLeft = (cx - 15, BORDER)
                # Draw a gray line showing the threshold for an "unknown" person
                ptThreshold = (ptTopLeft[0], ptBottomRight[1] - cv2.cv.Round((1.0 - UNKNOWN_PERSON_THRESHOLD) * faceHeight))
                cv2.rectangle(src, ptThreshold, (ptBottomRight[0], ptThreshold[1]), cv2.cv.CV_RGB(200,200,200), 1, cv2.CV_AA)
                # Crop the confidence rating between 0.0 to 1.0, to show in the bar.
                confidenceRatio =  1.0 - min(max(similarity, 0.0), 1.0)
                ptConfdence = (ptTopLeft[0], cv2.cv.Round(ptBottomRight[1] - confidenceRatio * faceHeight))
                # Show the light-blue confidence bar
                cv2.rectangle(src, ptConfdence, ptBottomRight, cv2.cv.CV_RGB(0,255,255), cv2.cv.CV_FILLED, cv2.CV_AA)
                # Show the gray border of the bar
                
                
                cv2.rectangle(src, ptTopLeft, ptBottomRight, cv2.cv.CV_RGB(200,200,200), 1, cv2.CV_AA)
                                
                textSize = cv2.getTextSize(outStr, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
                
                yOffset = 2
                textX = currFaceRect[0]  
                textY = currFaceRect[1] + textSize[0][1] + yOffset
                
                vertex = (currFaceRect[0] + textSize[0][0] , currFaceRect[1] + textSize[0][1] + yOffset + 4 )
                
                cv2.rectangle(src, (textX, textY - textSize[0][1] ), vertex, mFaceFrameColor, cv2.cv.CV_FILLED, cv2.CV_AA)
                cv2.putText(src,"{}".format(outStr), (textX, textY), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255), 2, bottomLeftOrigin=False)
#                 print identity
#                 print nameTranslations
                
                cv2.imshow('Video',src)
                
#                 cv2.waitKey(0)
#                 cv2.destroyAllWindows()
        elif mMode == MODE.DELETE_ALL:
            mSelectedPerson = 1
            mNumPersons = 0
            mLatestFaces = []
            preprocessedFaces = []
            faceLabels = []
            oldPreprocessedFace = None
            mMode = MODE.DETECTION
            testPreProcessedFaces = []
        else:
            print "ERROR: Invalid run mode {}".format(mMode)
            sys.exit()
            
    return None

def collectAndDetectFaces(folder, faceCascade, eyeCascade1, eyeCascade2, mode):
    global mMode,mDebug, preprocessLeftAndRightSeparately, preprocessedFaces, faceLabels, mNumPersons, nameTranslations, mPaintEyeCircle, mPaintFaceFrame, mFaceFrameColor, mEyeCircleColor, testPreProcessedFaces
    pic, label, nameTranslations = csv.getPhotoAndLabel(folder)
    for i in range(len(pic)):
        if i % 500 == 0:
            print "processed {} images".format(i)
        myPrint (pic[i])
#         print pic[i]
        img = cv2.imread(pic[i])
        
        preprocessedFace, faceRect, leftEye, rightEye, searchedLeftEye, searchedRightEye = getPreprocessedFace(img, faceWidth, faceCascade, eyeCascade1, eyeCascade2, preprocessLeftAndRightSeparately)
#         print preprocessedFace
        
        if preprocessedFace is not None and len(preprocessedFace) > 0:
            if mPaintFaceFrame:
                cv2.rectangle(img, (faceRect[0], faceRect[1]), (faceRect[0] + faceRect[2], faceRect[1] + faceRect[3]), mFaceFrameColor, 2, cv2.cv.CV_AA) # Check faceRect data
            myPrint(preprocessedFace, False)
            
            if mPaintEyeCircle:
                eyeColor = mEyeCircleColor
                '''
                -------------------------------------------------------------------
                '''
                radius = 6
    
                if leftEye is not None and leftEye[0] >= 0:
                    myPrint (leftEye)
                    leftEyeCenterX = cv2.cv.Round(faceRect[0]+leftEye[0])
                    leftEyeCenterY = cv2.cv.Round(faceRect[1]+leftEye[1] + 9)
                    cv2.circle(img, (leftEyeCenterX, leftEyeCenterY), radius, mEyeCircleColor) # Check circle for python
                if rightEye is not None and  rightEye[0] >= 0:
                    rightEyeCenterX = cv2.cv.Round(faceRect[0]+rightEye[0])
                    rightEyeCenterY = cv2.cv.Round(faceRect[1]+rightEye[1] + 9)
                    cv2.circle(img, (rightEyeCenterX, rightEyeCenterY), radius, mEyeCircleColor) # Check circle for python
                
#             cv2.imshow('{} - {}'.format(label[i], i),img)
            if mode == MODE.TEST:
#                 print "adding preprocessed test"
#                 testPreProcessedFaces.append(preprocessedFace)
#                 testFaceLables.append(label[i])
                pass
            else:
#                 print "adding preprocessed training"
                mirroredFace = cv2.flip(preprocessedFace, 1)
                preprocessedFaces.append(preprocessedFace)
                preprocessedFaces.append(mirroredFace)
                faceLabels.append(label[i])
                faceLabels.append(label[i])
        '''
        -------------------------------------------------------------------
        '''    
            
        
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    myPrint("preprocessedFaces = {}".format(len(preprocessedFaces)))
    mNumPersons = len(np.unique(faceLabels))
    myPrint(faceLabels)

def trainNetwork():
    global preprocessedFaces, faceLabels, facerecAlgorithm, mNumPersons, model, mTrainingTime
    haveEnoughData = True
    
    if facerecAlgorithm == "Fisherfaces":
        if mNumPersons < 2 or mNumPersons == 2 and mLatestFaces[1] < 0:
            print "Warning: Fisherfaces needs at least 2 people, otherwise there is nothing to differentiate! Collect more data."
            haveEnoughData = False
    
    if mNumPersons < 1 or len(preprocessedFaces) <= 0 or len(preprocessedFaces) != len(faceLabels):
        print "Warning: Need some training data before it can be learnt! Collect more data."
        haveEnoughData = False
        
    if haveEnoughData:
        tempTime = time.time()
        model = learnCollectedFaces(preprocessedFaces, faceLabels, facerecAlgorithm, model)
        mTrainingTime = time.time() - tempTime
        print "Training time with {} algorithm - {}".format(facerecAlgorithm, mTrainingTime)
        return model
    
        
#         if mDebug:
#                     showTrainingDebugData(model, faceWidth, faceHeight)
        
    pass

def getPersonName(n):
    global nameTranslations
    
    return nameTranslations.get(n)

def recognize(src, id, model, faceCascade, eyeCascade1, eyeCascade2):
    img = cv2.imread(src)
    preprocessedFace, faceRect, leftEye, rightEye, searchedLeftEye, searchedRightEye = getPreprocessedFace(img, faceWidth, faceCascade, eyeCascade1, eyeCascade2, preprocessLeftAndRightSeparately)
    gotFaceAndEyes = False
    recognitionSuccess = False
    if faceRect is not None:
        gotFaceAndEyes=True
    if gotFaceAndEyes and len(preprocessedFaces) > 0 and len(preprocessedFaces) == len(faceLabels):
        reconstructedFace = reconstructFace(model, preprocessedFace)
        if mDebug or True:
            if reconstructedFace is not None and len(reconstructedFace) > 0:
                cv2.imshow("reconstructedFace", reconstructedFace)
                
        # Verify whether the reconstructed face looks like the preprocessed face, otherwise it is probably an unknown person
        similarity = getSimilarity(preprocessedFace, reconstructedFace)
        
        
        if(similarity < UNKNOWN_PERSON_THRESHOLD):
            # Identify who the person is in the preprocessed face image.
#             print "IDENTIFY"
            identity = model.predict(preprocessedFace)
            outStr = getPersonName(identity[0]) 
            if id == identity[0]:
                recognitionSuccess = True
        else:
            # Since the confidence is low, assume it is an unknown person
            outStr = "Unknown"
        
        myPrint("Identity: {0}. Similarity: {1}".format(outStr, similarity))
        
        #Show the confidence rating for the recognition in the mid-top of the display
        cx = (len(img[0]) - faceWidth) / 2
        ptBottomRight = (cx - 5, BORDER + faceHeight)
        ptTopLeft = (cx - 15, BORDER)
        # Draw a gray line showing the threshold for an "unknown" person
        ptThreshold = (ptTopLeft[0], ptBottomRight[1] - cv2.cv.Round((1.0 - UNKNOWN_PERSON_THRESHOLD) * faceHeight))
#         print ptThreshold, ptBottomRight[0]
        cv2.rectangle(img, ptThreshold, (ptBottomRight[0], ptThreshold[1]), cv2.cv.CV_RGB(200,200,200), 1, cv2.CV_AA)
        # Crop the confidence rating between 0.0 to 1.0, to show in the bar.
        confidenceRatio =  1.0 - min(max(similarity, 0.0), 1.0)
        ptConfdence = (ptTopLeft[0], cv2.cv.Round(ptBottomRight[1] - confidenceRatio * faceHeight))
#         print ptBottomRight, ptConfdence
        # Show the light-blue confidence bar
        cv2.rectangle(img, ptConfdence, ptBottomRight, cv2.cv.CV_RGB(0,255,255), cv2.cv.CV_FILLED, cv2.CV_AA)
        # Show the gray border of the bar
        cv2.rectangle(img, ptTopLeft, ptBottomRight, cv2.cv.CV_RGB(200,200,200), 1, cv2.CV_AA)
        
        textSize = cv2.getTextSize(outStr,cv2.FONT_HERSHEY_DUPLEX,1.0,2)
        
        yOffset = 2
        textX = faceRect[0] + faceRect[2] - textSize[0][0] 
        textY = faceRect[1] + faceRect[3] - yOffset
        
        vertex = (faceRect[0] + faceRect[2], faceRect[1] + faceRect[3])
        
        cv2.rectangle(img, (textX, textY - textSize[0][1] - 1), vertex, cv2.cv.CV_RGB(0,0,0), cv2.cv.CV_FILLED, cv2.CV_AA)
        cv2.putText(img,"{}".format(outStr), (textX, textY), cv2.FONT_HERSHEY_DUPLEX, 1.0, mFaceFrameColor, 2, bottomLeftOrigin=False)
#         cv2.imshow('recognized face', img)
#         cv2.moveWindow('recognizedFace', 0,0)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
        return recognitionSuccess



def storeCollectedFaces():
    global preprocessedFaces, faceLabels, imgFolderPath, model,facerecAlgorithm
    if model is None:
        model = learnCollectedFaces(preprocessedFaces, faceLabels, facerecAlgorithm, model)
    
#     decide where the faces will go to
#     referenceMat = []
#     for label in (np.unique(faceLabels)):
#         model.predict(preprocessedFaces[])
#         continue
    myPath = imgFolderPath 
    for i in xrange(len(preprocessedFaces)):
        person = "{}.{}".format(faceLabels[i], nameTranslations.get(faceLabels[i]))
        img = preprocessedFaces[i] 
        imgName = time.time()
        newFolder = createNewFolder(myPath, faceLabels[i], nameTranslations.get(faceLabels[i]))
        pathname = "{}{}/pp/{}({}).jpg".format(myPath, person, imgName, i)
#         print pathname
#         cv2.cv.SaveImage(pathname, img)
        cv2.imwrite(pathname, img)
    
#     print("Saved {} images".format(len(faceLabels)))
    
def recognizeFromTestFolder(folder, faceCascade, eyeCascade1, eyeCascade2):
    global testPreProcessedFaces, model
    testPreProcessedFaces, labels, _ = csv.getPhotoAndLabel(folder)
#     print labels
#     collectAndDetectFaces(testFolderPath, faceCascade, eyeCascade1, eyeCascade2, MODE.TEST)
    totalTestPhotos = len(labels)
    totalRecogSuccess = 0
    for  i in  xrange(len(testPreProcessedFaces)):
        if recognize(testPreProcessedFaces[i], labels[i], model, faceCascade, eyeCascade1, eyeCascade2):
            totalRecogSuccess += 1
    
    print "Successful recognitions {}/{}".format(totalRecogSuccess, totalTestPhotos)

def getModeString(mode):
    if mode == 1:
        return "Detection"
    elif mode == 2:
        return "Face Collection"
    elif mode == 3:
        return "Training"
    elif mode == 4:
        return "Recognition"
    

def recognizeAndTrain(src, faceCascade, eyeCascade1, eyeCascade2):
    
    oldPreprocessedFace = None

    # Start in detection mode
    global mMode, mSelectedPerson, mNumPersons, model, modelPath, facerecAlgorithm, mReadFromFiles, mPaintEyeCircle, mPaintFaceFrame, camFrame
    
    mMode = MODE.DETECTION
    cam = cv2.VideoCapture(0)
    mSelectedPerson = 1
    centered = False
    
#     pool = ThreadPool(processes=1)
#     async_result = pool.apply(PopupWindow.askSaveAuto, (mReadFromFiles, model is not None))
    
    if runType == TYPE.PICTURE:
        # Run once for pictures
        # read csv file
        startTime = time.time()
        collectAndDetectFaces(imgFolderPath,faceCascade, eyeCascade1, eyeCascade2, MODE.TRAINING)
        print "finished in collecting faces in {} seconds".format(time.time() - startTime)
        # x is your dataset
#         indices = np.random.permutation(preprocessedFaces.shape[0])
#         training_idx, test_idx = indices[:]
#         preprocess, test = preprocessedFaces[:80,:], preprocessedFaces[80:,:]
        model = trainNetwork()
        recognizeFromTestFolder(testFolderPath, faceCascade, eyeCascade1, eyeCascade2)
#         recognize('/home/daniel/Desktop/Pics/Training/2/Felicia2.jpg', model, faceCascade, eyeCascade1, eyeCascade2)
#         recognize('/home/daniel/Documents/Untitled.jpeg', model, faceCascade, eyeCascade1, eyeCascade2)
    else:
        # Run forever until user hits esc in case it is video 
        while True:
            ret, frame = cam.read()
            
            if camFrame is None:
                camFrame = (frame.shape[1], frame.shape[0])
            
#             collectAndDetectFaces(faceCascade, eyeCascade1, eyeCascade2)
#             model = trainNetwork()
            oldPreprocessedFace = doStuff(frame, faceCascade, eyeCascade1, eyeCascade2, oldPreprocessedFace)
#             print oldPreprocessedFace
            
            modeStr = "Mode: {}".format(getModeString(mMode))
            modeTextSize = cv2.getTextSize(modeStr, cv2.FONT_HERSHEY_PLAIN, 2, 2)
            cv2.putText(frame,modeStr, (2 , camFrame[1] - 4), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2, bottomLeftOrigin=False)
            counter=1 
            
            if mMode != MODE.RECOGNITION:
                for num in  np.unique(faceLabels):
                    outStr = "collected faces for person {}: {} ".format(num, faceLabels.count(num))
                    textSize = cv2.getTextSize(outStr, cv2.FONT_HERSHEY_PLAIN, 1, 1)
                    cv2.putText(frame,outStr, (2 , (textSize[0][1]+4) * counter), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1, bottomLeftOrigin=False)
                    counter += 1
            
            # Print current person
            collectionStr = "Current Selected person: {}".format(getPersonName(mSelectedPerson))
            collectionTextSize = cv2.getTextSize(collectionStr, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            cv2.putText(frame, collectionStr,(camFrame[0] - collectionTextSize[0][0] - 4, collectionTextSize[0][1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1, bottomLeftOrigin=False)
            
            cv2.imshow('Video', frame)
            
            if not centered :
                    cv2.moveWindow('Video', screenWidth /3, 0)
                    centered = True
                    
            key = cv2.waitKey(1)
            
            # Quit
            if key & 0xFF == ord('q'):
                cam.release()
                cv2.destroyAllWindows()
                break
            
            # Standby mode, only detecting face
            if key & 0xFF == ord('d'):
                myPrint("Changed MODE to Detection", True)
                mMode = MODE.DETECTION
            
            # Collect faces 
            if key & 0xFF == ord('c'):
                myPrint("Changed MODE to Collect Faces", True)
                mMode = MODE.COLLECT_FACES
                
            # Run Training Mode
            if key & 0xFF == ord('t'):
                myPrint("Changed MODE to Training", True)
                mNumPersons = len(np.unique(faceLabels))
                mMode = MODE.TRAINING
            
            # Run Recognition Mode
            if key & 0xFF == ord('r'):
                myPrint("Changed MODE to Recognition", True)
                mMode = MODE.RECOGNITION
            
            # Delete in memory data (preprocessed faces, labels, etc) 
            if key & 0xFF == ord('x'):
                myPrint("DELETE ALL", True)
                mMode = MODE.DELETE_ALL
            
            # Train from existing photo files    
            if key & 0xFF == ord('f'):
                myPrint("Train from Files", True)
                mReadFromFiles = True
                collectAndDetectFaces(imgFolderPath, faceCascade, eyeCascade1, eyeCascade2, MODE.TRAINING)
                model = trainNetwork()
                mMode = MODE.RECOGNITION    
            # Save collected faces
            if key & 0xFF == ord('s'):
                
#                 print async_result
#                 return_val = async_result.get()
#                 print return_val
#                 thread = Thread(target=PopupWindow.askSaveAuto(mReadFromFiles, model is not None))
#                 thread.start()
#                 thread.join()
                
                print("Storing collected faces")
                storeCollectedFaces()
                mMode = MODE.DETECTION
            
            # Store model data in a file
            if key & 0xFF == ord('m'):
                if model is not None:
                    print("Storing training model data.")
                    model.save(modelPath)
                else:
                    print("There is no trained model data to save.")
            
            # Change current person       
            if key & 0xFF >= ord('1') and key & 0xFF <= ord('9') :
                mSelectedPerson = int(chr(key & 0xFF))
                myPrint("Changed person to {}".format(mSelectedPerson), True)
            
            # Toggle Paint eye cirlce
            if key & 0xFF == ord('e'):
                mPaintEyeCircle = not mPaintEyeCircle
            
            # Toggle Paint face frame
            if key & 0xFF == ord('w'):
                mPaintFaceFrame = not mPaintFaceFrame
            
            # Load from existing Model file
            if key & 0xFF == ord('l'):
                try:
                    if model is None:
                        print("Loading training model data.")
                        if facerecAlgorithm == "Fisherfaces":
                            model = cv2.createFisherFaceRecognizer()
                        elif facerecAlgorithm == "Eigenfaces":
                            model = cv2.createEigenFaceRecognizer()
                        else:
                            model = cv2.createLBPHFaceRecognizer() 
                        
                    
                    model.load(modelPath)
                except Exception as e:
                    print("There is no model data to load or an error occured while loading.")
                    
            
        
run()
        