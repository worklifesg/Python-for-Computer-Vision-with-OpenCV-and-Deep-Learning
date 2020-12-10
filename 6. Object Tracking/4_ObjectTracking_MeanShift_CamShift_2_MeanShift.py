import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret,frame = cap.read()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_rects = face_cascade.detectMultiScale(frame) 

#track one single only using this code

(face_x,face_y,w,h)=tuple(face_rects[0]) 

track_window = (face_x,face_y,w,h) #to track the tuple

#Region of interest

roi = frame[face_y:face_y+h,face_x:face_x+w]

#HSV color mapping

hsv_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

#ROI histogram to back track mean shift

roi_hist = cv2.calcHist([hsv_roi],[0],None,[180],[0,180])

cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

#Determinent criteria

term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,1)

while True:
  ret, frame = cap.read() #keep grabbing from camera

  if ret == True:
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

    ret, track_window = cv2.meanShift(dst,track_window,term_criteria)

    x,y,w,h = track_window
    img2 = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),5) #cont. updating the rectangle

    cv2.imshow('img',img2)
    
    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break
  else:
    break

cv2.destroyAllWindows()
cap.release() 
