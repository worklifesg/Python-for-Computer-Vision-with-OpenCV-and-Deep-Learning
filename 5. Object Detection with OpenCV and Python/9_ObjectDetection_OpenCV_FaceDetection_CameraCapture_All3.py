#Writing a program to detect face from live camera feed

#Import Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

#upload 3 images and in gray scale
nadia = cv2.imread('Nadia_Murad.jpg',0)
denis = cv2.imread('Denis_Mukwege.jpg',0)
solvay = cv2.imread('solvay_conference.jpg',0)

# Pre-trained XML Haarcascade files

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
# Functions to detect face and eyes
def detect(img):

  face_img = img.copy()
  face_rects = face_cascade.detectMultiScale(face_img)

  for (x,y,w,h) in face_rects:
    cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,130,0),10)
  
  eyes_rects = eye_cascade.detectMultiScale(face_img,scaleFactor=1.2,minNeighbors=5)

  for (x,y,w,h) in eyes_rects:
    cv2.rectangle(face_img,(x,y),(x+w,y+h),(0,180,60),10)

  smile_rects = smile_cascade.detectMultiScale(face_img,1.7,20)

  for (x,y,w,h) in smile_rects:
    cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,0,130),10)
  
  return face_img
# Real live camera feed and detection

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read(0)
    frame = detect(frame) #change detect_eyes for eyes detection

    cv2.imshow('Video Face Detect',frame)

    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
