'''
Created on Aug 31, 2016

@author: daniel
'''

class Point:
    '''
    Repreents and manipulates x,y coords.
    '''


    def __init__(self, x = 0, y = 0):
        '''
        Create a new point at x, y
        '''
        self.x = x
        self.y = y
        
    def __str__(self):
        return "({0}, {1})".format(self.x, self.y)