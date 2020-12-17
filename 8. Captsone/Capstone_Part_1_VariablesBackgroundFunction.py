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
