# Here we will use callbacks to conenct iamges to event functions with OpenCV (allows us to directly interact with images)

 # Connecting Callback functions
 # Adding Functionality through Event Choices
 # Dragging mouse for functionality
# connect function to drawing

import cv2
import numpy as np

###############
## VARIABLES ##
###############

drawing = False #Will be True if the mouse has been pressed down
ix = -1
iy = -1

###############
## FUNCTION ##
###############

def draw_rectangle(event,x,y,flags,params): #these parameters are called back automatically with MousecallBack
  global ix,iy,drawing
  if event == cv2.EVENT_LBUTTONDOWN: #for starting drawing rectangle
      drawing = True
      ix,iy = (x,y)
  elif event == cv2.EVENT_MOUSEMOVE: #for dragging to keep the shape of rectangle
      if drawing == True:
          cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1) #ix,iy starting point and dragging point, x,y are where current position is

  elif event == cv2.EVENT_LBUTTONUP: #to finish drawing final rectangle
      drawing = False
      cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)

################################
## SHOWING IMAGE WITH OPENCV ###
################################

img = np.zeros((512,512,3)) #int8 gives a bit gray background

cv2.namedWindow(winname='My_Drawing')
cv2.setMouseCallback('My_Drawing',draw_rectangle)

while True:
  cv2.imshow('My_Drawing',img)
  if cv2.waitKey(1) & 0xFF == 27:
    break

cv2.destroyAllWindows()
