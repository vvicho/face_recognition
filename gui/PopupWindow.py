'''
Created on Oct 6, 2016

@author: daniel
'''

import easygui as eg
from Tkinter import *
import cv2.cv as cv
import Tkinter
# print (eg.choicebox("hola", "que te gusta", arr1))
# print (eg.enterbox("Title", "Text"))

def askNewPerson():
    
    return eg.ynbox("Is the person new to the system?","")

# print askNewPerson()

def newPerson():
    return eg.enterbox("Please type the name of this person", "Register New Person")

# print newPerson()

# Show the application keyboard options
def showOptions():
    optionString = "Please note the following commands:\n\
    d - Detect Mode\n\
    c - Collect Data Mode\n\
    t - Train Network with available data\n\
    r - Recognition Mode\n\
    f - Load and train network from existing Photos in database\n\
    m - Save trained network\n\
    l - Load previously saved trained network\n\
    s - Save photos\n\
    x - Delete all in memory data\n\
    q - Exit application\n\
    0-9 - Choose person\n\
    \
    Paint Options:\n\
    e - Draw eye circles\n\
    r - Draw face frame"
    
    return eg.msgbox(optionString, "title")

# showOptions()
    
def askSaveAuto(readFromFiles=False, trained=False):
    if readFromFiles and trained:
        msg = "Do you wish to save these pictures automatically?\nYou might be asked if a person is new to the system or if the network fails to recognize someone 80% of the times."
        return eg.buttonbox(msg,"Save Pictures?", choices=('Yes', 'No', 'Cancel'))
    elif trained:
        msg = "Do you wish to save these pictures automatically?\nThe network must be retrained to classify the pictures in the correct folders. You might be asked if a person is new to the system or if the network fails to recognize someone 80% of the times."
        return eg.buttonbox(msg,"Save Pictures?", choices=('Yes', 'No', 'Cancel'))
    else:
        msg = "Do you wish to save the new pictures automatically? This might take some time since the network needs to be trained. You might be asked if a person is new to the system or if the network fails to recognize someone 80% of the times." 
        return eg.buttonbox(msg,"Save Pictures?", choices=('Yes', 'No', 'Cancel'))    

# print askSaveAuto()

def askSamePerson():
    eg.indexbox("Are these the same Daniel?", "Same Person?", image='/home/daniel/Desktop/Pics/Training/1.Daniel/beard.jpg')
    
    
# askSamePerson()

# def buttonScreen():
#     root = Tk()
#     w = Canvas(root, width=500, height=300, bd=10, bg='white')
#     w.grid(row=0, column=0, columnspan=2)
#     
#     b = Button(width=10, height=2, text='Button1')
#     b.grid(row=1, column=0)
#     b2 = Button(width=10, height=2, text='Button2')
#     b2.grid(row=1, column=1)
#     
#     cv.NamedWindow("camera",1)
#     capture = cv.CaptureFromCAM(0)
#     
#     while True:
#         img = cv.QueryFrame(capture)
#         w.create_image(0,0,image=img)
#         if cv.WaitKey(10) == 27:
#             break
#     
#     root.mainloop()
# 
# buttonScreen()
    