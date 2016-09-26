'''
Created on Sep 21, 2016

@author: daniel
'''

import cv2

cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    cv2.imshow('bla', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
    
    