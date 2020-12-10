import cv2
import numpy as np

#Creating a dictionary to track corners
corner_track_params = dict(maxCorners=10,
                           qualityLevel=0.3,
                           minDistance=7,
                           blockSize=7)
#parameters for Lucas Kanade function
#Smaller window - more senstive to noise and may miss larger motions
'''
LK method using the idea of image pyramid 
Level0 - original image, Level1 - 1/2 resol, Level2 - 1/4 resol, Level3 - 1/8 resol, Level4 - 1/15 resol
At each level the image is blurred and subsample i.e. allows to find optical flow at various resolutions
'''
lk_params = dict(winSize=(200,200),
                 maxLevel=2,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))
# live streaming capturing of Sparse Optical Flow

cap = cv2.VideoCapture(0)
ret,prev_frame = cap.read()

prev_gray = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)

# points to track

prevPts = cv2.goodFeaturesToTrack(prev_gray,mask=None,**corner_track_params)

#mask
mask = np.zeros_like(prev_frame)

while True:

  ret, frame = cap.read()
  frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

  nextPts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray,frame_gray,prevPts, None,**lk_params)

  good_new = nextPts[status==1]
  good_prev = prevPts[status==1]

  for i, (new,prev), in enumerate(zip(good_new,good_prev)):
    x_new,y_new = new.ravel()
    x_prev,y_prev = prev.ravel()

    mask = cv2.line(mask,(x_new,y_new),(x_prev,y_prev),(0,255,0),3)

    frame = cv2.circle(frame,(x_new,y_new),8,(0,0,255),-1) #draing circle on current points in a frame are
  
  img = cv2.add(frame,mask)
  cv2.imshow('tracking',img)

  k = cv2.waitKey(30) & 0xFF
  if k == 27:
    break
  
  prev_gray = frame_gray.copy()
  prevPts = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()
