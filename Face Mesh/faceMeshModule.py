import cv2 as cv 
import mediapipe as mp 


class FaceMeshDector():
    def __init__(self, num_face=1):
        self.num_face = num_face

        self.mpfaceMesh = mp.solutions.face_mesh
        self.mpDraw = mp.solutions.drawing_utils
        self.faceMesh = self.mpfaceMesh.FaceMesh(max_num_faces= self.num_face)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1,circle_radius=1 , color=(255,255,0))

    def findFaceMesh(self, img):
        imgRGB = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        self.results = self.faceMesh.process(imgRGB)

        if self.results.multi_face_landmarks:
            for landmks in self.results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img,landmks, self.mpfaceMesh.FACEMESH_CONTOURS, self.drawSpec,self.drawSpec)

                for id,lm in enumerate(landmks.landmark):
                    #print(lm)
                    ih, iw, ic = img.shape
                    x,y = int(lm.x*iw), int(lm.y*ih)
                    print(id,x,y)



def main():

    cap = cv.VideoCapture(0)
    face = FaceMeshDector()

    while True:
        success, img = cap.read()
        face.findFaceMesh(img)
        cv.imshow("face mesh", img)
        if cv.waitKey(20) & 0xff == ord("q"):
            break

if __name__ == "__main__":
    main()