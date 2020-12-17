############################
'''
In this project, we will create a program that can detect a hand, 
segment the hand and count the number of fingers being held up

In PART I:
 - Define global variables
 - Function that updates running average of the background values in an ROI

Strategy for counting the fingers:
### Grab ROI
### Calculate running average background for 60 frames of video
### Once average value is found, then the hand can enter ROI an apply thresholding etchniques to apply isolation
### Once the hand enters ROI we will use COnvex Hull to draw polygon around hand
### Calculate center of hand and calculate angle of center with the outer fingers points
### Also using some angles we can calculate some distances between fingers.
'''

# Import libraries
import cv2
import numpy as np

from sklearn.metrics import pairwise #distance calculation

## Global variables

background=None 
accumulated_weight=0.5 #[0-1]

roi_top = 20 # setting ROI dimensions on camera
roi_bottom = 300
roi_right = 300
roi_left = 600

#Function for avg background value

def calc_accum_avg(frame,accumulated_weight):

    global background #first time it is None

    if background is None:
        background = frame.copy().astype('float')
        return None
    
    cv2.accumulateWeighted(frame,background,accumulated_weight)
    # The function calculates the weighted sum of the input image src and the accumulator dst so that dst
    # # becomes a running average of a frame sequence:
    ### Not returning anything in this function, just gloabl variable and updating accumulated weight function.

#################################################################################################################
################################################################################################################
 ### PART II - SEGMENTATION
#need to check this threshold value for different objects
def segment(frame,threshold_min=25):
     
     diff = cv2.absdiff(background.astype('uint8'),frame) #absolute difference

     ret,thresholded = cv2.threshold(diff,threshold_min,255,cv2.THRESH_BINARY) #min value 25

     contours,hierarchy = cv2.findContours(thresholded.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)# to grab the image and contours

     if len(contours) == 0: # if we didn't grab any contours
         return None
    

     else:
         #Assuming largest external contour in ROI is HAND
         hand_segment = max(contours,key=cv2.contourArea)

         return(thresholded,hand_segment)

#################################################################################################################
################################################################################################################
 ### PART III - COUNTING FINGERS AND CONVEXHULL

def count_fingers(thresholded,hand_segment):

    conv_hull = cv2.convexHull(hand_segment)

    #EXTREME POINTS 
    top = tuple(conv_hull[conv_hull[:,:,1].argmin()][0]) #most extreme top point
    bottom= tuple(conv_hull[conv_hull[:,:,1].argmax()][0])
    left = tuple(conv_hull[conv_hull[:,:,0].argmin()][0])
    right = tuple(conv_hull[conv_hull[:,:,0].argmax()][0])

    cX = (left[0]+right[0])// 2
    cY = (top[1]+bottom[1])// 2

    distance = pairwise.euclidean_distances([[cX,cY]],[left,right,top,bottom])[0] #diatance between fingers

    max_distance = distance.max()

    #Create a circle around max distance

    radius = int(0.9*max_distance) #90% of max distance
    circumfrence = (2*np.pi*radius)

    circular_roi = np.zeros(thresholded.shape[:2],dtype='uint8')

    cv2.circle(circular_roi,(cX,cY),radius,255,10)

    circular_roi = cv2.bitwise_and(thresholded,thresholded,mask=circular_roi)

    #grab contours in ROI

    contours,hierarchy = cv2.findContours(circular_roi.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    count = 0 #no figners are extended

    for cnt in contours:

        (x,y,w,h) = cv2.boundingRect(cnt)
        #to check contour is not at very bottom region
        out_of_wrist = (cY+(cY*0.25)) > (y+h)

        #doesn't exceed 25  of points outside

        limit_points = ((circumfrence*0.25) > cnt.shape[0])

        if out_of_wrist and limit_points:
            count+=1
            
    return count

#################################################################################################################
################################################################################################################

cam = cv2.VideoCapture(0)

num_frames = 0 
#first 60 frames for background avg

while True:
    ret,frame = cam.read()
    frame_copy = frame.copy()

    roi  =frame[roi_top:roi_bottom,roi_right:roi_left]
    #gray scale
    gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    # apply bit of blur to gray scale image
    gray = cv2.GaussianBlur(gray,(7,7),0)

    #accumulated weights
    if num_frames <60:
        calc_accum_avg(gray,accumulated_weight)

        if num_frames <=59:
            cv2.putText(frame_copy,'WAIT, GETTING BACKGROUND',(200,200),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(0,0,255),2)
            cv2.imshow('Finger Count',frame_copy)
    else:

        hand = segment(gray)

        if hand is not None:
            thresholded,hand_segment = hand

            #DRAW contours around real hand in live stream
            cv2.drawContours(frame_copy,[hand_segment+(roi_right,roi_top)],-1,(255,0,0),5)

            fingers = count_fingers(thresholded,hand_segment)

            cv2.putText(frame_copy,str(fingers),(70,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

            cv2.imshow('Thresholded', thresholded)

    cv2.rectangle(frame_copy,(roi_left,roi_top),(roi_right,roi_bottom),(0,0,255),5)

    num_frames +=1

    cv2.imshow('Finger Count', frame_copy)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()
