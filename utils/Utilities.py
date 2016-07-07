'''
Created on Jun 29, 2016

@author: daniel
'''
import os
import random
from random import shuffle
from utils import Photo as photo_obj

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
#             person_name = person.split('/')
#             photo = photo_obj.Photo(person + f, person_name[len(person_name)-1], f )
#             photo.path = person + f
#             photo.album_name = person_name[len(person_name)-1]
#             photo.photo_name = f
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

def write_file(str_array):
    temp_path = "logs/file.txt" 
    #Set new_text to whatever you want based on your logic
    buf = "\n".join(str_array)    
    f=open(temp_path,'w')
    f.write(buf)
    f.close();
    
    

    