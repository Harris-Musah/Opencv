import cv2 as cv 
import time 
import numpy as np 
import HandsTrackingModule as htm 
import math

from ctypes import cast,POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#########################
wCam,hCam = 640,480
#########################

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_,CLSCTX_ALL,None)

volume = cast(interface,POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

# video capture
cap = cv.VideoCapture(0)
cap.set(3,wCam)
cap.set(4, hCam)
ptime = 0

detector = htm.HandDetector(detectionCon=0.7)
volBar = 400
volPer = 0
while True:
    succ, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)
    if len(lmlist) != 0:
        #print(lmlist[4], lmlist[8])
        x1,y1 = lmlist[4][1],lmlist[4][2]
        x2,y2 = lmlist[8][1],lmlist[8][2]
        cx,cy =(x1+x2)//2, (y1+y2)//2

        cv.circle(img,(x1,y1), 15,(255,0,255),cv.FILLED)
        cv.circle(img,(x2,y2), 15,(255,0,255),cv.FILLED)
        #circle at the center 
        cv.circle(img,(cx,cy), 15,(255,0,255),cv.FILLED)
        cv.line(img,(x1,y1),(x2,y2),(255,0,255), 3)

        #calculating length
        length = math.hypot(x2-x1,y2-y1)
        #print(length)
        #hand range 50 to 300
        # volume range -65 to 0

        vol = np.interp(length,[50,300],[minVol, maxVol])
        volBar = np.interp(length,[50,300],[400, 150])
        volPer = np.interp(length,[50,300],[0, 100])
        volume.SetMasterVolumeLevel(vol,None)
        # print(vol)

        if length < 50:
            cv.circle(img, (cx,cy), 15, (0,255,0), cv.FILLED)

        
    cv.rectangle(img,(50,155),(85,400),(0,255,0), 3)
    cv.rectangle(img,(50,int(volBar)),(85,400),(0,255,0), cv.FILLED)
    cv.putText(img, f'{int(volPer)}%', (40,450) ,cv.FONT_HERSHEY_COMPLEX_SMALL,2,(255,0,255),1)

    ctime = time.time()
    fps =  1 /(ctime-ptime)
    ptime = ctime 
    cv.putText(img, f'fps:{int(fps)}%', (40,70) ,cv.FONT_HERSHEY_COMPLEX_SMALL,2,(0,255,0),1)
    cv.imshow("vid", img)
    if cv.waitKey(20) & 0xff == ord('d'):
        break