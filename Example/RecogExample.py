#!/usr/bin/python

# Import the required modules
import cv2
import os
import numpy as np
from PIL import Image
from utils import Utilities as uti

face_size = 70

# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "/home/daniel/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# For face recognition we will the the LBPH Face Recognizer 
recognizer = cv2.createFisherFaceRecognizer()

def get_images_and_labels(path):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad')]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    for image_path in image_paths:
        # Read the image and convert to grayscale
        image_pil = Image.open(image_path).convert('L')
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        
        # Get the label of the image
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        # Detect the face in the image
        faces = faceCascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            faceImg= image[y: y + h, x: x + w]
            faceImg = cv2.resize(faceImg, (face_size, face_size))
            images.append(faceImg)
            labels.append(nbr)
#             cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
#             cv2.waitKey(50)
    # return the images list and labels list
    return images, labels

def get_images_and_labels2(path, train=True):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    images_ = uti.getAllPhotos(path)
    for img in images_:
        if train :
            if '001' in img.path:
                continue
            # Read the image and convert to grayscale
            image_pil = Image.open(img.path).convert('L')
            # Convert the image format into numpy array
            image = np.array(image_pil, 'uint8')
            # Get the label of the image
            print(img.album_name)
            nbr = int(img.album_name) 
            #int(os.path.split(img.path)[1].split(".")[0].replace("subject", ""))
            # Detect the face in the image
            faces = faceCascade.detectMultiScale(image)
            # If face is detected, append the face to images and the label to labels
            for (x, y, w, h) in faces:
                images.append(image[y: y + h, x: x + w])
                labels.append(nbr)
    #             cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
    #             cv2.waitKey(50)
        else:
            if '001' not in img.path:
                continue
            # Read the image and convert to grayscale
            image_pil = Image.open(img.path).convert('L')
            # Convert the image format into numpy array
            image = np.array(image_pil, 'uint8')
            # Get the label of the image
            nbr = int(img.album_name) #int(os.path.split(img.path)[1].split(".")[0].replace("subject", ""))
            # Detect the face in the image
            faces = faceCascade.detectMultiScale(image)
            # If face is detected, append the face to images and the label to labels
            for (x, y, w, h) in faces:
                images.append(image[y: y + h, x: x + w])
                labels.append(nbr)
            
    # return the images list and labels list
    return images, labels

# Path to the Yale Dataset
# path = '/home/daniel/workspace/Project/Images/yalefaces/jpeg/'
path = '/home/daniel/workspace/Project/Images/Test/1/'




# Call the get_images_and_labels function and get the face images and the 
# corresponding labels
images, labels = get_images_and_labels(path)
for n in images:
#     print (n)
    print "x = {} y = {} pixels = {}".format(len(n), len(n[0]), len(n) * len(n[0]))
    pass
cv2.destroyAllWindows()

# Perform the tranining
recognizer.train(images, np.array(labels))
print recognizer.getMat("eigenvectors")
help(recognizer)
print(recognizer.__str__())
print dir(recognizer)


# Append the images with the extension .sad into image_paths
image_paths = uti.getAllPhotos(path)
for image_path in image_paths:
#     if '001.' not in image_path.path:
#         continue
    predict_image_pil = Image.open(image_path.path).convert('L')
    print(image_path.path)
    predict_image = np.array(predict_image_pil, 'uint8')
    print(predict_image)
    faces = faceCascade.detectMultiScale(predict_image)
    for (x, y, w, h) in faces:
#         nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
        nbr_predicted = recognizer.predict(predict_image[y: y + h, x: x + w])
        nbr_actual = image_path.album_name #int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        if nbr_actual == nbr_predicted:
#             print ("{0} is Correctly Recognized with confidence {}".format(nbr_actual, conf))
            print ("{0} is Correctly Recognized with confidence".format(nbr_actual))
        else:
            print ("{0} is Incorrect Recognized as {1}".format(nbr_actual, nbr_predicted))
        cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
        cv2.waitKey(1000)