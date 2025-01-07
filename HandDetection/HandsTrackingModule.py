import cv2 as cv
import time
import mediapipe as mp 


class HandDetector():
    def __init__(self, mode= False,maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,max_num_hands= self.maxHands,min_detection_confidence=self.detectionCon,min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, image, draw=True):
        imageRBG = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRBG)

        if self.results.multi_hand_landmarks:
            for handlmks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, handlmks, self.mpHands.HAND_CONNECTIONS)

        return image

    def findPosition(self, image, handNo= 0,draw= True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myhand.landmark):
                #print(id,lm)
                h,w, ch = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv.circle(image,(cx,cy), 15, (255,0,255), cv.FILLED)

        return lmlist


def main():

    cap = cv.VideoCapture(0)

    pTime = 0
    cTime = 0
    detector = HandDetector()
    while True:
        success, frames = cap.read()
        img = detector.findHands(frames)
        lmlist = detector.findPosition(frames)
        if len(lmlist) != 0:
            print(lmlist[4])

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv.putText(frames,str(int(fps)),(10,70), cv.FONT_HERSHEY_COMPLEX,3,(255,0,255),2)
        cv.imshow("hands", frames)
        if  cv.waitKey(20) & 0xff == ord('d'):
            break



if __name__ == "__main__":
    main()