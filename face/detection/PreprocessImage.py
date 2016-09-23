'''
Created on Sep 2, 2016

@author: daniel
'''

# import cv2
import cv2 
import utils.DetectObject
from utils.DetectObject import detectLargestObject
from math import atan2
import numpy as np
from cv2 import imshow

DESIRED_LEFT_EYE_X = 0.16
DESIRED_LEFT_EYE_Y = 0.14
FACE_ELLIPSE_CY = 0.40
FACE_ELLIPSE_W = 0.50
FACE_ELLIPSE_H = 0.80


'''
 Search for both eyes within the given face image. Returns the eye centers in 'leftEye' and 'rightEye',
 or sets them to (-1,-1) if each eye was not found. Note that you can pass a 2nd eyeCascade if you
 want to search eyes using 2 different cascades. For example, you could use a regular eye detector
 as well as an eyeglasses detector, or a left eye detector as well as a right eye detector.
 Or if you don't want a 2nd eye detection, just pass an uninitialized CascadeClassifier.
 Can also store the searched left & right eye regions if desired.
'''
def detectBothEyes(face, eye_cascade1, eye_cascade2, left_eye, right_eye,
                   searched_left_eye, searched_right_eye):
    
    leftEye = rightEye = None
    
    # For default eye.xml or eyeglasses.xml: Finds both eyes in roughly 40% of detected faces, but does not detect closed eyes
    eye_sx = 0.16
    eye_sy = 0.26
    eye_sw = 0.30
    eye_sh = 0.28
    
    cols = len(face[0])
    rows = len(face)    
    
    leftX = cv2.cv.Round(cols * eye_sx)
    topY = cv2.cv.Round(rows * eye_sy)
    widthX = cv2.cv.Round(cols * eye_sw)
    heightY = cv2.cv.Round(rows * eye_sh)
    rightX = cv2.cv.Round(cols * (1.0 - eye_sx - eye_sw)) # start of right eye corner
    
    topLeftOfFace = face[topY:topY + heightY, leftX:leftX + widthX]
    topRightOfFace = face[topY:topY + heightY, rightX:rightX + widthX]
    
#     leftEyeRect = rightEyeRect = None
    
    
    # Return the search windows to the caller, if desired
    searched_left_eye = [leftX, topY, widthX, heightY]
    searched_right_eye = [rightX, topY, widthX, heightY]
    
    # Search the left region, then the right region using the 1st eye detector
    leftEyeRect = detectLargestObject(topLeftOfFace, eye_cascade1, cols, "detectBothEyes - cascade1 - leftEye")
    rightEyeRect = detectLargestObject(topRightOfFace, eye_cascade1, cols, "detectBothEyes - cascade1 - rightEye")
    
    # if the eye is not detected, try another classifier
    if len(leftEyeRect <= 0 and eye_cascade2 != None):
        leftEyeRect = detectLargestObject(topLeftOfFace, eye_cascade2, cols, "detectBothEyes - cascade2 - leftEye")
    
    #Same as above, try another classifier if the eye is not detected.
    if len(rightEyeRect <= 0 and eye_cascade2 != None):
        rightEyeRect = detectLargestObject(topRightOfFace, eye_cascade2, cols, "detectBothEyes - cascade2 - rightEye")
    
    if len(leftEyeRect) > 0:    # Check if the eye was detected
        leftEyeRect[0] += leftX # Adjust the left-eye rectangle because the face border was removed
        leftEyeRect[1] += topY
        leftEye = None          #Point(leftEyeRect[0] + leftEyeRect[2]/2.0, leftEyeRect[1] + leftEyeRect[3]/2.0)
    else:
        leftEye = (-1, -1) # return an invalid point
    
    if len(rightEyeRect) > 0:       # Check if the eye was detected
        rightEyeRect[0] += rightX   # Adjust the left-eye rectangle because the face border was removed
        rightEyeRect[1] += topY
        rightEye = None             #(rightEyeRect[0] + rightEyeRect[2]/2.0, rightEyeRect[1] + rightEyeRect[3]/2.0)
    else:
        rightEye = (-1, -1)
    
    height, width = face.shape[:2]
    
    
    
    
    
    return leftEye, rightEye, searched_left_eye, searched_right_eye
    
print(round(02.365))

'''
 Histogram Equalize seperately for the left and right sides of the face.
'''
def equalizeLeftAndRightHalves(faceImg) :

#  It is common that there is stronger light from one half of the face than the other. In that case,
#  if you simply did histogram equalization on the whole face then it would make one half dark and
#  one half bright. So we will do histogram equalization separately on each face half, so they will
#  both look similar on average. But this would cause a sharp edge in the middle of the face, because
#  the left half and right half would be suddenly different. So we also histogram equalize the whole
#  image, and in the middle part we blend the 3 images together for a smooth brightness transition.
    w = len(faceImg[0])
    h = len(faceImg)
    
    # 1) Equalize the whole face
    wholeFace = equalizeLeftAndRightHalves(faceImg)
    
    # 2) Equalize the left half and right half of the face separately
    midX = w / 2.0
    leftSide = faceImg[0:h, 0:midX]
    rightSide = faceImg[0:h, midX:w]
    
    leftSide = cv2.equalizeHist(leftSide)
    rightSide = cv2.equalizeHist(rightSide)
    
    # 3) Combine the left half and right hald and whole face together, so that it has a smooth transition
    
    for y in range(h):
        for x in range(w):
            if x < w / 4.0: # Left 25%: just use the left face
                v = leftSide[y,x]
            elif x < w* 2.0/4.0: # Mid-left 25%: blend the left face & whole face
                lv = leftSide[y,x]
                wv = wholeFace[y,x]
                # Blend more of the whole face as it moves further right along the face
                f = (x - w * 1.0/4.0) / (w * 0.25)
                v = cv2.cv.Round((1.0 - f) * lv + (f) * wv)
            elif x < w * 3.0/4.0: # Mid-right 25%: blenf the right face & whole face
                rv = rightSide[y, x-midX]
                wv = wholeFace[y, x]
                # Blend more of the right-side face as it moves further right along the face
                f = (x - w * 2.0/4.0) / (w * 0.25)
                v = cv2.cv.Round((1.0 - f) * wv + (f) * rv)
            else: # Right 25%: just use the right face
                v = rightSide[y, x-midX]
                
            faceImg[y,x] = v
        #end x loop            
    #end y loop 

    return faceImg


'''
 Create a grayscale face image that has a standard size and contrast & brightness
 srcImg should be a cpoy of the whole color camera frame or picture so it can draw the eye positions
 if 'doLeftAndRightSeparately' is True, it will process left and right eyes separately,
 si that if there is a strong light on one side but not the other, it will still look OK.
 Performs Face Preprocessing as a combination of:
  - geometrical scaling, rotation and translation using Eye Detection,
  - smoothing away image noise using a Bilateral Filter,
  - standardize the brightness on both left and right sides of the face independently using separated Histogram Equalization,
  - removal of background and hair using an Elliptical Mask.
 Returns either a preprocessed face square image or NULL (ie: couldn't detect the face and 2 eyes).
 If a face is found, it can store the rect coordinates into 'storeFaceRect' and 'storeLeftEye' & 'storeRightEye' if given,
 and eye search regions into 'searchedLeftEye' & 'searchedRightEye' if given.
'''
def getPreprocessedFace(srcImg, desiredFaceWidth, faceCascade, eyeCascade1, eyeCascade2, doLeftAndRightSeparately=True
                        , storeFaceRect=None, storeLeftEye=None, storeRightEye=None
                        , searchedLeftEye=None, searchedRightEye=None):
#     variable[x, y, w, h]
#     storedFaceRect = [0,0,0,0]
#     storedLeftEye = [0,0,0,0]
#     storedRightEye = [0,0,0,0]
#     searchedLeftEye = [0,0,0,0]
#     searchedRightEye = [0,0,0,0]
   
    # use squared faces
    desiredFaceHeight = desiredFaceWidth
    
    # Mark the detected face region and eye search regions as invalid, in case they aren't detected
    if storeFaceRect:
        storeFaceRect[2] = -1
    if storeLeftEye:
        storeLeftEye[0] = -1
    if storeRightEye:
        storeFaceRect[0] = -1
    if searchedLeftEye:
        searchedLeftEye[2] = -1
    if searchedRightEye:
        searchedRightEye[2] = -1
    
    faceRect = detectLargestObject(img=srcImg, cascade=faceCascade,largestObject=(20,20), details='getProcessedFace')
    
    # Check if a face was detected
    if len(faceRect) > 0:
        
        #store the face
        if storeFaceRect.size:
            storeFaceRect = faceRect
        
        # Get the rect details
        x,y,w,h = faceRect[0], faceRect[1], faceRect[2], faceRect[3]
        # Get the detected face image
        faceImg = srcImg[y:y+h, x:x+w]
        
        # If the input image is not grayscale, convert the bgr or bgra color image to grayscale
        # Converted images (or gray) return with only to values from shape function. 
        if len(faceImg.shape) > 2:
            gray = cv2.cvtColor(faceImg, cv2.COLOR_BGR2GRAY)
        else :
            gray = faceImg 

        # Search for the 2 eyes at the full resolution, since eye detection needs max resolution possible
        # leftEye, rightEye = None, None
        
        leftEye, rightEye, searchedLeftEye, searchedRightEye = detectBothEyes(gray, eyeCascade1, eyeCascade2, searchedLeftEye, searchedRightEye)
        
        if storeLeftEye:
            storeLeftEye = leftEye
        if storeRightEye:
            storeRightEye = rightEye
        
        #Check if both eyes were detected
        if leftEye[0] >= 0 and rightEye[0] >= 0:
            # Make the face image the same size as the training images
            
            eyesCenter = ((leftEye[0]+rightEye[0]) / 2.0, (leftEye[1] + rightEye[1]) / 2.0)
            # get the angle between the 2 eyes
            dy = rightEye[1] - leftEye[1]
            dx = rightEye[0] - leftEye[0]
            leng = (dx*dx + dy*dy) ** 0.5
            angle = atan2(dy,dx) * 180.0 / cv2.cv.CV_PI # Convert from radians to degrees
            
            # hand measurements shown that the left eye center should ideally be at roughly (0.19, 0.14) of a scaled face image.
            DESIRED_RIGHT_EYE_X = (1.0 - DESIRED_LEFT_EYE_X)
            
            # Get the amount we need to scale the image to be the desired fixed size we want
            desiredLen = (DESIRED_RIGHT_EYE_X - DESIRED_LEFT_EYE_X) * desiredFaceWidth            
            scale = desiredLen / leng
            
            # Get the transformation matrix for rotating and scaling the face to the desired angle and size
            rot_mat = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
            # Shift the center of the eyes to be the desired center between the eyes
            rot_mat[0,2] += desiredFaceWidth * 0.5 - eyesCenter[0]
            rot_mat[1,2] += desiredFaceHeight * DESIRED_LEFT_EYE_Y - eyesCenter[1]
            
            # Rotate and scale and translate the image to the desired angle, size and position
            # Note that we use w for the height instead of h, because the input face has 1:1 aspect ratio.
            
            warped = np.array(desiredFaceHeight, desiredFaceWidth, cv2.CV_8U, cv2.cv.Scalar(128))
            warped = cv2.warpAffine(gray, rot_mat, len(warped.size))
            
            cv2.imshow('warped', warped)
            
            # Give the image a standard brightness and contrast in case it was too dark or had low contrast.
            if not doLeftAndRightSeparately:
                # Do it on the whole face
                warped = cv2.equalizeHist(warped)
            else:
                # Do it separately for the left and right sides of the face
                equalizeLeftAndRightHalves(warped)
                
            cv2.imshow('equalized', warped)
            
            #Use the bilateral filter to reduce pixel noise by smoothing the image, but keeping the sharp edges in the face
            filtered = np.array(warped.shape[0:1], cv2.CV_8U)
            filtered = cv2.bilateralFilter(warped, 0, 20.0, 2.0)
            cv2.imshow('filtered', filtered)
            
            #Filter out the corners of the face, since we mainly just care about the middle parts.
            # Draw a filled ellipse in the middle of the face-sized image
            
            mask = np.array(warped.shape[0:1], cv2.CV_8U, cv2.cv.Scalar(0)) # Empty mask
            faceCenter = (desiredFaceWidth / 2.0, cv2.cv.Round(desiredFaceHeight * FACE_ELLIPSE_CY))
            size = (cv2.cv.Round(desiredFaceWidth * FACE_ELLIPSE_W), cv2.cv.Round(desiredFaceHeight * FACE_ELLIPSE_H))
            cv2.ellipse(mask, faceCenter, size, 0, 0, 0, 360, cv2.cv.Scalar(255), 1, cv2.cv.CV_FILLED) # number 1 after scalar indicates thickness
            
            cv2.imshow('mask', mask) 
            
            dstImg = np.zeros(warped.shape[0:1], cv2.CV_8U)
            cv2.cv.Copy(filtered, dstImg, mask)
            
            imshow('dstImg', dstImg)
            
            return dstImg, storeFaceRect, storeLeftEye, storeRightEye, searchedLeftEye, searchedRightEye
            
            
            
            
                
    # Mark the detected face region and eye search regions as invalid, in case they aren't detected
#     if
return np.zeros() 



'''
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
    '' !!!!!!!!!!!   ''
    ey = DESIRED_FACE_HEIGHT * DESIRED_LEFT_EYE_X - eyes_center.y 
    '' !!!!!!!!!!!   ''
    
#     rot_mat.at(0,2) += ex
#     rot_mat.at(1,2) += ey 
    rot_mat[0,2] += ex
    rot_mat[1,2] += ey 
    
    # Transform the face image to the desired angle & size & position!
    # Also clear the transformed image background to a default grey
    warped = np.mat((DESIRED_FACE_HEIGHT, DESIRED_FACE_WIDTH), np.uint8)
    
    cv2.warpAffine(gray, warped, rot_mat, len(warped))
    
'''