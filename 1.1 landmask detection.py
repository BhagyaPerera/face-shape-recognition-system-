import cv2
import dlib
from imutils import face_utils

camera=cv2.VideoCapture(0)
face_detector=dlib.get_frontal_face_detector()
#a pretrained face detecting classifier in dlib module   

landmark_detector=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#pretrained external algorithm for detecting 68 points of a face in a image





while(True):

     ret,img=camera.read()
   




     if(ret==True):
        img[50:100,50:100]=[255,0,0]


        
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        rects=face_detector(gray)
        #alogorithm.predict

        for rect in rects:

            x1=rect.left()
            y1=rect.top()
            x2=rect.right()
            y2=rect.bottom()

            print(x1,y1,x2,y2)

            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1-1,y1-30),(x1+60,y1),(0,255,0),-1)
            
            #img-where the rect should be drwan
            #(0,255,0)-color in BGR
            #2-line width in pixel
            cv2.putText(img,'FACE',(x1+2,y1-2),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)

            points=landmark_detector(gray,rect)
            #passing the gray image and bounding rectangle to the landmark detector
            #points object contains the 68 points
            points=face_utils.shape_to_np(points)
            #converting the 68 points object into numpy array



            for p in points:
                
                cen=(p[0],p[1])
                cv2.circle(img,cen,2,(0,255,255),-1)


            
            
        
        cv2.imshow('IMG',img)   
        cv2.imshow('GRAY',gray)
        #cv2.imshow('RECT',rects)
        
        cv2.waitKey(1)
        
