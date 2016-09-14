'''
Created on Jul 6, 2016

@author: daniel
'''
import cv2
import sys

haarcascades_path = '/home/daniel/opencv/data/haarcascades/'
FACES_PATH = '/home/daniel/Downloads/Images/lfw-deepfunneled/'

# cascPath = haarcascades_path
faceCascade = cv2.CascadeClassifier(haarcascades_path + 'haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier(haarcascades_path + 'haarcascade_eye.xml')
right_eye_cascade = cv2.CascadeClassifier(haarcascades_path + 'haarcascade_lefteye_2splits.xml')
left_eye_cascade =  cv2.CascadeClassifier(haarcascades_path + 'haarcascade_righteye_2splits.xml')
profile_face_cascade = cv2.CascadeClassifier(haarcascades_path + 'haarcascade_profileface.xml')
nose_cascade = cv2.CascadeClassifier(haarcascades_path + 'Nariz.xml')
# smile_cascade =  cv2.CascadeClassifier(haarcascades_path + 'haarcascade_smile.xml')

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

#     profile_faces = faceCascade.detectMultiScale(gray,
#         scaleFactor=1.1,
#         minNeighbors=3,
#         minSize=(30, 30),
#         flags=cv2.CASCADE_SCALE_IMAGE)
    roi_color = None
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(30,30))
        right_eyes = right_eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(30,30))
        left_eyes = left_eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(30,30))
        noses = nose_cascade.detectMultiScale(roi_gray)
#         smiles = smile_cascade.detectMultiScale(roi_gray)
#         for(ex, ey, ew, eh) in eyes:
#             cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255,0,0),2)
#         for(ex, ey, ew, eh) in noses:
#             cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,0,255),2)
        
        for(ex, ey, ew, eh) in right_eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255,0,0),2)
#         for(ex, ey, ew, eh) in left_eyes:
#             cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,255),2)
        #Paint Mouths
#         for(ex, ey, ew, eh) in smiles:
#             cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255,127,200),2)

#     for (x, y, w, h) in profile_faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = frame[y:y+h, x:x+w]

    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    #Display Face only
    if roi_color == None:
        cv2.destroyWindow('Face')
    else:
        cv2.imshow('Face', roi_color)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
