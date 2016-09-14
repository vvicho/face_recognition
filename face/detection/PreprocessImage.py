'''
Created on Sep 2, 2016

@author: daniel
'''

# import cv2
import cv2 
import utils.DetectObject
from utils.DetectObject import detectLargestObject

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
def detectBothEyes(face, cascade_classifier, eye_cascade, left_eye, right_eye,
                   searched_left_eye, searched_right_eye):
    
    eye_sx = 0.16
    eye_sy = 0.26
    eye_sw = 0.30
    eye_sh = 0.28
    
    height, width = face.shape[:2]
    
    
print(round(02.365))


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
def getPreprocessedFace(srcImg, desiredFaceWidth, faceCascade, eyeCascade1, eyeCascade2, doLeftAndRightSeparately=True, storeFaceRect, storeLeftEye, storeRightEye, searchedLeftEye, searchedRightEye):
    # variable[x, y, w, h]
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
        leftEye, rightEye = None, None
        '''
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        DETECTBOTHEYES
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        '''
        
    # Mark the detected face region and eye search regions as invalid, in case they aren't detected
    if 



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