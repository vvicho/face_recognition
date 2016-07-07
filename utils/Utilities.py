'''
Created on Jun 29, 2016

@author: daniel
'''
import os
import random
from random import shuffle

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
    
    listOfPeople = getTrimmedList(getSubdirectories(direc), numOfPeople )

    photos = []
    for person in listOfPeople:
        
        photos += [person + f for f in getRandomFile(person, picPerPerson)] 
    
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
    print(photos)
#     print(len(photos))
    

def getAllPhotos(direc):
    return getRandomImages(direc, -1, -1)
    

# def getPhoto(photos):
#     

test()