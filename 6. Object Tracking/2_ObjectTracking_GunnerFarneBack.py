# Gunner Farneback's algorithm

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, frame1 = cap.read() #capturing image

prvsImg = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) #converting to grayscale

hsv_mask = np.zeros_like(frame1)
hsv_mask[:,:,1] = 255

while True:
    ret,frame2 = cap.read()
    #compare previous entire iamge to the current image
    nextImg = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    #optical flow
    #calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
    flow = cv2.calcOpticalFlowFarneback(prvsImg,nextImg,None,0.5,3,15,3,5,1.2,0) #default
    
    #Flow object contains vector flow cartesian information - pointing at direction each pixel is happening in.
    #convert to poalr (mag, angle) that can be mapped to hue color mapping
    # mag - saturation direction, angle between hue and saturation
    #moving left will be red, moving red - blue
    
    mag, ang = cv2.cartToPolar(flow[:,:,0],flow[:,:,1],angleInDegrees=True) #for every pixel in x,y pick horiztonal and vertical
    
    hsv_mask[:,:,0] = ang/2 #looking at half the hues (180-360)
    hsv_mask[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    
    bgr = cv2.cvtColor(hsv_mask,cv2.COLOR_HSV2BGR)
    
    cv2.imshow('frame',bgr)
    
    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break
    prvsImg = nextImg

cv2.destroyAllWindows()
cap.release()  
    
