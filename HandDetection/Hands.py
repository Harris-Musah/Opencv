import cv2 as cv
import time
import mediapipe as mp 

cap = cv.VideoCapture(0)
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHands.Hands()

pTime = 0
cTime = 0

while True:
    success, frames = cap.read()
    imageRBG = cv.cvtColor(frames, cv.COLOR_BGR2RGB)
    results = hands.process(imageRBG)
    
    if results.multi_hand_landmarks:
        for handlmks in results.multi_hand_landmarks:
            for id, lm in enumerate(handlmks.landmark):
               # print(id,lm)
                h,w, ch = frames.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                if id == 4:
                    cv.circle(frames,(cx,cy), 15, (255,0,255), cv.FILLED)

            mpDraw.draw_landmarks(frames, handlmks, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv.putText(frames,str(int(fps)),(10,70), cv.FONT_HERSHEY_COMPLEX,3,(255,0,255),2)
    cv.imshow("hands", frames)
    if  cv.waitKey(20) & 0xff == ord('d'):
        break