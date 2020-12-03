#TASK: (NOTE: YOU WILL NEED TO RUN THIS AS A SCRIPT). 
# 
# Create a script that opens the picture and allows you to draw empty red circles whever you click the RIGHT MOUSE BUTTON DOWN.

#import libraries
import cv2
import numpy as np

###############
## FUNCTION ##
###############

def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(fix_img,(x,y),50,(0,0,255),thickness=10)

# Connects the mouse button to our callback function
cv2.namedWindow(winname='Dog_backpack')
cv2.setMouseCallback('Dog_backpack',draw_circle)


#Runs forever until we break with Esc key on keyboard
    # Shows the image window
    # EXPLANATION FOR THIS LINE OF CODE:
    # https://stackoverflow.com/questions/35372700/whats-0xff-for-in-cv2-waitkey1/39201163


################################
## SHOWING IMAGE WITH OPENCV ###
################################

img = cv2.imread('dog_backpack.png')
fix_img = cv2.resize(img,(0,0),img,0.5,0.5)

while True:
    cv2.imshow('Dog_backpack',fix_img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
    
# Once script is done, its usually good practice to call this line
# It closes all windows (just in case you have multiple windows called)
cv2.destroyAllWindows()
