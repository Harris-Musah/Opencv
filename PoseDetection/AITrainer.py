import cv2 as cv 
import numpy as np 
import time 

cap = cv.VideoCapture(0)

while True:
    success, img = cap.read()


    cv.imshow('vid', img)
    if cv.waitKey(20) & 0xff == ord('d'):
        break