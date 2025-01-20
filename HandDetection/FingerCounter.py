import cv2 as cv
import os 
import HandsTrackingModule as htm

cap =  cv.VideoCapture(0)

folderPath = 'C:/Users/DELL/Documents/New folder/Murtaza/Opencv/HandDetection/images'
mylist = os.listdir(folderPath)
#print(mylist)

overlaylist = []

for imgPath in mylist:
    image = cv.imread(f'{folderPath}/{imgPath}')
    #print(f'{folderPath}/{imgPath}')
    overlaylist.append(image)

#print(len(overlaylist))

detector = htm.HandDetector()

tipIds = [4,8,12,16,20]


while True:
    success, img = cap.read()
    img = detector.findHands(img)

    lmlist = detector.findPosition(img,draw=False)
    #print(lmlist)

    if len(lmlist) !=0:
        fingers = []
        #thumb
        if lmlist[tipIds[0]][1] > lmlist[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        #4 fingers
        for id in range(1,5):
            if lmlist[tipIds[id]][2] < lmlist[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        #print(fingers)
        totalFingers = fingers.count(1)
        #print(totalFingers)

        h,w,c = overlaylist[totalFingers-1].shape
        img[0:h, 0:w] = overlaylist[totalFingers-1]

    cv.imshow('vid',img)
    if cv.waitKey(20) & 0xff ==ord("d") :
        break