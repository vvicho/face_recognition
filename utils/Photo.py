'''
Created on Jun 29, 2016

@author: daniel
'''

class Photo:
    '''
    classdocs
    '''


    def __init__(self, path, album_name, photo_name):
        '''
        Constructor
        '''
        self.path = path
        self.album_name = album_name
        self.photo_name = photo_name
        
    def __repr__(self):
        return self.photo_name