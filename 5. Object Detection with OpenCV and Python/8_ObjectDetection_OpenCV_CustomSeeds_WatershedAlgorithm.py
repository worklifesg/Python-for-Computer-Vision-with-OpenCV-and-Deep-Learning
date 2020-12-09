import cv2
import numpy as np
import matplotlib.pyplot as plt

road = cv2.imread('road_image.jpg')

scale_percent = 100 # percent of original size
width = int(road.shape[1] * scale_percent / 100)
height = int(road.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
road = cv2.resize(road, dim, interpolation = cv2.INTER_AREA) 
road_copy = np.copy(road)

#set marker image
marker_image = np.zeros(road.shape[:2],dtype=np.int32)
segments = np.zeros(road.shape,dtype=np.uint8)

from matplotlib import cm

#we want this to be tuple

def create_rgb(i):
  return tuple(np.array(cm.tab10(i)[:3])*255)

#defining its RGB colors
colors=[]
for i in range(10):
  colors.append(create_rgb(i))

######################
## GLOBAL VARIABLES ##
n_markers = 10 #0-9
current_marker=1 #color choice

marks_updated = False #markers updated by Watershed

######################
## CALLBACK FUNCTION ##

def mouse_callback(event,x,y,flags,param):
  global marks_updated

  if event == cv2.EVENT_LBUTTONDOWN:
    #Markers paased to watershed algo
    cv2.circle(marker_image,(x,y),2,(current_marker),-1)
    #user sees on image
    cv2.circle(road_copy,(x,y),2,colors[current_marker],-1)

    marks_updated = True

######################
## WHILE TRUE ##

cv2.namedWindow('Road Image')
cv2.setMouseCallback('Road Image',mouse_callback)

while True:
  cv2.imshow('Watershed Segments',segments)
  cv2.imshow('Road Image',road_copy)

  #Close all windows

  k = cv2.waitKey(1)
  if k == 27:
    break

  #Clearing all colors (USER PRESSES C KEY)

  elif k == ord('c'):
    road_copy = road.copy()
    marker_image = np.zeros(road.shape[:2],dtype=np.int32)
    segments = np.zeros(road.shape,dtype=np.uint8)

  #update color choice

  elif k > 0 and chr(k).isdigit():
    current_marker = int(chr(k)) #check current marker

  # Update markings

  if marks_updated:
    marker_image_copy = marker_image.copy()
    cv2.watershed(road,marker_image_copy)

    segments = np.zeros(road.shape,dtype=np.uint8)

    for color_ind in range(n_markers):
      segments[marker_image_copy == (color_ind)] = colors[color_ind]

cv2.destroyAllWindows()
