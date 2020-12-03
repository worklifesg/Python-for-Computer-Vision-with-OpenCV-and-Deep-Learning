# Here we will use callbacks to conenct iamges to event functions with OpenCV (allows us to directly interact with images)

 # Connecting Callback functions
 # Adding Functionality through Event Choices
 # Dragging mouse for functionality
# connect function to drawing

import cv2
import numpy as np

###############
## FUNCTION ##
###############

def draw_circle(event,x,y,flags,param): #these parameters are called back automatically with MousecallBack
  #pass #automatically filled
  if event == cv2.EVENT_LBUTTONDOWN:
    cv2.circle(img,(x,y),50,(0,255,0),-1)

cv2.namedWindow(winname='My_Drawing')
cv2.setMouseCallback('My_Drawing',draw_circle)

################################
## SHOWING IMAGE WITH OPENCV ###
################################

img = np.zeros((512,512,3),np.int8) #int8 gives a bit gray background

while True:
  cv2.imshow('My_Drawing',img)
  if cv2.waitKey(20) & 0xFF == 27:
    break

cv2.destroyAllWindows()
