import cv2 as cv 
import mediapipe as mp  
 

cap =  cv.VideoCapture(0)
mpFaceDettection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDettection.FaceDetection()
while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    #print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            mpDraw.draw_detection(img, detection)
            #print(id, detection.location_data.relative_bounding_box)


    cv.imshow("vid", img)
    if cv.waitKey(20) & 0xff == ord("d"):
        break