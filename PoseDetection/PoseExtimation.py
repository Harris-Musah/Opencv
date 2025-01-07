import cv2 as cv
import mediapipe as mp 


cap = cv.VideoCapture(0)
mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
Pose = mpPose.Pose()


while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = Pose.process(imgRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img,results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            cx ,cy = lm.x*h, lm.y*w
            
    
    cv.imshow("vid", img)
    if cv.waitKey(20) & 0xff == ord("d"):
        break