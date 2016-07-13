'''
Created on Jun 29, 2016

@author: daniel
'''
import os
import random
from random import shuffle
from utils import Photo as photo_obj
import cv2

def getSubdirectories(direc):
    '''
    Gets all subdirectories in direc path
    Returns a list of directory names
    '''

    out = []
    for _, dirnames, _ in os.walk(direc):
        # Get all subdirectories from the Face library
        out = dirnames
        break
        # Return full path of the people directories
    return [direc + name + "/" for name in out]
        
        
def getRandomFile(direc, quantity=1):
    '''
    Gets quantity random files from direc directory 
    Returns a list with quantity file names
    '''
    # Boolean to check if all 
    allPhotos = quantity == -1
    
    for _, _, filenames in os.walk(direc):
        shuffle(filenames)
        
        # Return all filenames if quantity is -1 
        if allPhotos:
            return filenames
        
        # If quantity is set to more than the photos that exist in the directory
        # Return all the  fileNames
        
        if(quantity > len(filenames)):
            return filenames 
        
        # Else return a sublist from the first element to the "quantity"th 
        return filenames[:quantity]

        
def getTrimmedList(listA, length):
    '''
    Gets a list objects, shuffles them and returns a shorter list of lenght elements.
    If length value is greater than list size, it returns the shuffled initial list
    '''
    
    # If length is -1 we reutrn the whole shuffled List of people
    if length == -1:
        shuffle(listA)
        return listA
    
    else:
        # If people list size is less than the desired length, we return the shuffled input list 
        if len(listA) <= length:
            shuffle(listA)
            listB = listA
        else:
            # Return a shuffled sublist of people of size:length 
            shuffle(listA)
            listB = listA[:length]
        
        return listB


def getRandomImages(direc, numOfPeople, picPerPerson):
    '''
    Gets images from 
    '''
#     print(numOfPeople, picPerPerson)
#     print(direc)
    listOfPeople = getTrimmedList(getSubdirectories(direc), numOfPeople )

    photos = []
    for person in listOfPeople:
        for f in getRandomFile(person, picPerPerson):
            photo = createPhoto_obj(person, f)
            photos += [photo]
#         photos += [person + f for f in getRandomFile(person, picPerPerson)] 
#         photo = photo_obj()
        
    return photos

def test():
    random.seed(None)
    haarcascades_path = '/home/daniel/opencv/data/haarcascades/'
    FACES_PATH = '/home/daniel/Downloads/Images/lfw/'
    listA = getTrimmedList(getSubdirectories(FACES_PATH), 7)
#     print(listA)
#     print(len(listA))
    photos = getRandomImages(FACES_PATH, 2, 4)
#     photos = getAllPhotos(FACES_PATH)
#     print(photos)
#     print(len(photos))
    

def createPhoto_obj(folder, name):
    trimmedPath = folder.split()
    return photo_obj.Photo(folder + name, trimmedPath[len(trimmedPath)-1], name)

def getAllPhotos(direc):
    return getRandomImages(direc, -1, -1)
    



def getPhotosFrom(path):
    for _, _, filenames in os.walk(path):
        filenames.sort()
        return [createPhoto_obj(path, f) for f in filenames]
     

# test()

def write_file(filename, str_array):
#     temp_path = "logs/file.txt"
    buf = ""
    for photo in str_array:
        buf += photo.path + "\n"
        
    f = open("/home/daniel/workspace/Project/logs/{0}".format(filename),'w')
    f.write(buf)
    f.close();
    
    
# Photo Utils
def resize_img(img, longest_side=500):
    height, width = img.shape[:2]
    factor = 0
    if height > width:
        factor = (500*100/height)/100
    else:
        factor = (500*100/width)/100
    res = cv2.resize(img,None,fx=factor, fy=factor, interpolation = cv2.INTER_CUBIC)
    return res