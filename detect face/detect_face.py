import cv2
import numpy as np
import matplotlib as plt
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def detect_faces(image):
    face_img=image.copy()
    gray_img=cv2.cvtColor(face_img,cv2.COLOR_BGR2GRAY)
    face_rect=face_cascade.detectMultiScale(gray_img,scaleFactor=1.1,
                                            minNeighbors=5)
    for (x,y,w,h) in face_rect:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,0,0),2)
    return face_img

def detect_eyes(image):## eyes haarcascade file is not good for extended use you can change the file
    face_eyes=image.copy()
    gray_eyes=cv2.cvtColor(face_eyes,cv2.COLOR_BGR2GRAY)
    eye_cascade=cv2.CascadeClassifier("haarcascade_eye.xml")
    eyes=eye_cascade.detectMultiScale(gray_eyes,scaleFactor=1.1,
                                     minNeighbors=5)
    for (x,y,w,h) in eyes:
        cv2.rectangle(face_eyes,(x,y),(x+w,y+h),(0,255,0),2)
    return face_eyes
cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    if not ret:
        break
    faces=detect_faces(frame)
    cv2.imshow("Face Detection",faces)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break