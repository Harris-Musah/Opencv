import cv2 as cv 
import mediapipe as mp 

cap = cv.VideoCapture(0)

mpfaceMesh = mp.solutions.face_mesh
mpDraw = mp.solutions.drawing_utils
faceMesh = mpfaceMesh.FaceMesh()
drawSpec = mpDraw.DrawingSpec(thickness=1,circle_radius=1 , color=(255,255,0))

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for landmks in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img,landmks, mpfaceMesh.FACEMESH_CONTOURS, drawSpec,drawSpec)
            for id,lm in enumerate(landmks.landmark):
                #print(lm)
                ih, iw, ic = img.shape
                x,y = int(lm.x*iw), int(lm.y*ih)
                print(id,x,y)

    cv.imshow("face mesh", img)
    if cv.waitKey(20) & 0xff == ord("q"):
        break