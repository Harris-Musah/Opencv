import cv2 as cv
import mediapipe as mp 


class PoseDetector():
    def __init__(self, mode=False, smoothness=True, detectionCon=0.5,trackCon=0.5):
        self.mode = mode
        #self.ubberBody = ubberBody
        self.smoothness = smoothness
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.Pose = self.mpPose.Pose(static_image_mode=self.mode,smooth_landmarks=self.smoothness,min_detection_confidence=self.detectionCon,min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.Pose.process(imgRGB)

        if draw:
            if self.results.pose_landmarks:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img 

    def getPosition(self, img, draw=True):
        lmlist = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                cx ,cy = int(lm.x*w), int(lm.y*h)
                lmlist.append([id, cx,cy])
                if draw:
                    cv.circle(img,(cx,cy),5,(255,0,0), cv.FILLED)
        return lmlist
            
    

def main():
    cap = cv.VideoCapture(0)
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmlist = detector.getPosition(img, False)
        #print(lmlist)
        cv.imshow("vid", img)
        if cv.waitKey(20) & 0xff == ord("d"):
            break

if __name__ == "__main__":
    main()