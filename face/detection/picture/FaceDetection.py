'''
Created on Jun 23, 2016

@author: daniel
'''

import cv2
import numpy as np
import utils.Utilities as uti
import utils
import time

start_time = time.time()

# Directory paths
haarcascades_path = '/home/daniel/opencv/data/haarcascades/'
FACE_LIBRARY = 2
FACE_LIBRARIES = [
      '/home/daniel/Downloads/Images/lfw/',                 # 0
      '/home/daniel/Downloads/Images/lfw-deepfunneled/',    # 1
      '/home/daniel/Downloads/Images/Daniel/',              # 2 My personal photos             
      '/home/daniel/workspace/Project/Images/yalefaces/jpeg/subject07'    # 3 Yale Face Library
                 ]

FACE_DIRECTORY = FACE_LIBRARIES[FACE_LIBRARY]

# Number of photos configuration
people = -1
pic_per_person = -1 # -set to -1 to bring all photos from a person

# Face Detection coniguration
scale_factor = 1.11
num_neighbors = 3

# other configurations
print_photos = True
write_log = True
resize = True

face_cascade = cv2.CascadeClassifier(haarcascades_path + 'haarcascade_frontalface_alt2.xml')
eyeglasses_cascade = cv2.CascadeClassifier(haarcascades_path + 'haarcascade_eye_tree_eyeglasses.xml')
eye_cascade = cv2.CascadeClassifier(haarcascades_path + 'haarcascade_eye.xml')
right_eye_cascade = cv2.CascadeClassifier(haarcascades_path + 'haarcascade_lefteye_2splits.xml')
left_eye_cascade =  cv2.CascadeClassifier(haarcascades_path + 'haarcascade_righteye_2splits.xml')
smile_cascade =  cv2.CascadeClassifier(haarcascades_path + 'haarcascade_smile.xml')
fullbody_cascade = cv2.CascadeClassifier(haarcascades_path + 'haarcascade_fullbody.xml')

def getImages(filename=None, folder=None, num_of_people=people, num_of_pics=pic_per_person):
    if FACE_LIBRARY == 3:
        imgs = uti.getFiles(FACE_DIRECTORY, people)
        print(FACE_DIRECTORY)
        
    else:
    
        if(filename == None and folder == None):
                imgs = uti.getRandomImages(FACE_DIRECTORY, people, pic_per_person)
                # imgs = uti.getAllPhotos(FACES_PATH)
        else:
            if(folder==None):
                print("Get photo: " + filename)
                person_name = filename.split("_0")
                imgs = [uti.createPhoto_obj(FACE_DIRECTORY + person_name[0] + "/", filename )]
            else:
                print("Get photos from folder " + folder)
                imgs = uti.getPhotosFrom(FACE_DIRECTORY + folder + "/")
                
    return imgs


def detect(filename=None, folder=None, num_of_people=people, num_of_pics=pic_per_person):
    
    imgs = getImages(filename, folder, num_of_people, num_of_pics)
    # print(imgs)
    totalPhotos=len(imgs)
    photosWithNoFaces = []
    photosWithMoreThanOneFace = []
    photosWithMoreThan2Eyes = []
    index=1
    
    for photo in imgs:
        if index % 50 == 0:
            print(index)
    
        index +=1
        img = cv2.imread(photo.path)
        if resize:
            res = uti.resize_img(img, 500)
        else:
            res = img
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        
        
        
        
        faces = face_cascade.detectMultiScale(gray, scale_factor, num_neighbors)
    
        
        '''
        faces = face_cascade.detectMultiScale(gray, 1.9, 3)
        Total Photos 13233
        Photos with no faces: 3333
        Photos with more than one face : 341
        ----- 54.146432399749756 seconds ----
        *************************************************************
        faces = face_cascade.detectMultiScale(gray, 1.5, 3)
        Total Photos 13233
        Photos with no faces: 918
        Photos with more than one face : 5658
        ----- 91.67998313903809 seconds ----
        *************************************************************
        faces = face_cascade.detectMultiScale(gray, 1.3, 3)
        Total Photos 13233
        Photos with no faces: 418
        Photos with more than one face : 356
        ----- 128.8101842403412 seconds ----
        *************************************************************
        faces = face_cascade.detectMultiScale(gray, 1.1, 3)
        Total Photos 13233
        Photos with no faces: 68
        Photos with more than one face : 841
        ----- 257.9152545928955 seconds ----
        *************************************************************
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        Total Photos 13233
        Photos with no faces: 124
        Photos with more than one face : 558
        ----- 257.44470167160034 seconds ----
        '''
    #     print(faces)
        if(len(faces) ==0):
            photosWithNoFaces += [photo]
        elif(len(faces)>1):
            photosWithMoreThanOneFace += [photo]
    
#         print(photo.photo_name)
        if print_photos:
            for(x,y,w,h) in faces:
                #Paint face
                cv2.rectangle(res, (x,y), (x+w, y+h), (255,0,0),2)
                
                #Get all facial data
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = res[y:y+h, x:x+w]
                left_eyes = left_eye_cascade.detectMultiScale(roi_gray)
                right_eyes = right_eye_cascade.detectMultiScale(roi_gray)
                eyes = eye_cascade.detectMultiScale(roi_gray)
                smiles = smile_cascade.detectMultiScale(roi_gray)
                bodies = fullbody_cascade.detectMultiScale(roi_gray)
                
                #Paint both eyes
                for(ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,255),2)
                #Paint left eyes
                for(ex, ey, ew, eh) in left_eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0),2)
                #Paint right eyes
                for(ex, ey, ew, eh) in right_eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,0,255),2)
                #Paint Mouths
                for(ex, ey, ew, eh) in smiles:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255,127,200),2)
                #Paint bodies
                for(ex, ey, ew, eh) in bodies:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255,127,0),2)
                print("Eyes : {0}".format(len(eyes)))
                print("Left eyes : {0}".format(len(left_eyes)))
                print("Right eyes : {0}".format(len(right_eyes)))
        #     print("finished")  
              
            cv2.imshow(photo.photo_name, res)
            cv2.waitKey(0)
    cv2.destroyAllWindows()
#     print(type(["hola"]), type("hola"))
    
    if write_log:
        uti.write_file("photos_with_no_faces.txt", photosWithNoFaces)

    print("Total Photos " + str(totalPhotos))
    print("Photos with no faces: {0}".format((len(photosWithNoFaces))))
    print("Photos with more than one face : {0}".format(len(photosWithMoreThanOneFace)))
    
    total_time = time.time() - start_time
    print("----- %s seconds ----" %total_time)

# detect(folder="George_W_Bush")
# detect(filename='George_W_Bush_0011.jpg')
detect()