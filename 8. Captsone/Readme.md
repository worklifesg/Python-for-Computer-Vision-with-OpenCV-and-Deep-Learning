### Capstone Project

In this project, we will create a program that can detect a hand, 
segment the hand and count the number of fingers being held up

Strategy for counting the fingers:
* Grab ROI
* Calculate running average background for 60 frames of video
* Once average value is found, then the hand can enter ROI an apply thresholding etchniques to apply isolation
* Once the hand enters ROI we will use COnvex Hull to draw polygon around hand
* Calculate center of hand and calculate angle of center with the outer fingers points
* Also using some angles we can calculate some distances between fingers.
