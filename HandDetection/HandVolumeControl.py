import cv2 as cv 
import time 
import numpy as np 
import HandsTrackingModule as htm 

#########################
wCam,hCam = 640,480
#########################



cap = cv.VideoCapture(0)
cap.set(3,wCam)
cap.set(4, hCam)
ptime = 0

detector = htm.HandDetector(detectionCon=0.7)

while True:
    succ, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)
    if len(lmlist) != 0:
        print(lmlist[4], lmlist[8])



    ctime = time.time()
    fps =  1 /(ctime-ptime)
    ptime = ctime 
    cv.putText(img, f'fps:{int(fps)}', (40,70) ,cv.FONT_HERSHEY_COMPLEX_SMALL,2,(255,0,255),1)
    cv.imshow("vid", img)
    if cv.waitKey(20) & 0xff == ord('d'):
        break