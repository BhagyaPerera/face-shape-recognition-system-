import cv2  
import dlib
import imutils    
from imutils import face_utils

face_detector=dlib.get_frontal_face_detector()
landmark_detector=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


from sklearn.externals import joblib

clsfr=joblib.load('KNN_Model.sav')


def predict_face_type(points):

    label_dict={0:'Diamond',1:'Oblong',2:'Oval',3:'Round',4:'Square',5:'Triangle'}
    
    my_points=points[2:9,0]

    D1=my_points[6]-my_points[0]
    D2=my_points[6]-my_points[1]
    D3=my_points[6]-my_points[2]
    D4=my_points[6]-my_points[3]
    D5=my_points[6]-my_points[4]
    D6=my_points[6]-my_points[5]

    d1=D2/float(D1)*100
    d2=D3/float(D1)*100
    d3=D4/float(D1)*100
    d4=D5/float(D1)*100
    d5=D6/float(D1)*100

    result=clsfr.predict([[d1,d2,d3,d4,d5]])
    label=result[0]
    #print(label)
    text='FACE TYPE: '+label_dict[label]
    cv2.putText(img,text,(40,40),cv2.FONT_HERSHEY_SIMPLEX,1.3,(255,255,255),2)
    
camera=cv2.VideoCapture(0)
#camera object has the access to the default camera of your pc

while(True):

    ret,img=camera.read()
    #img is a single frame (RGB) captured by the camera
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #img will be converted into a gray image

    rect=face_detector(gray)

    (height,width)=img.shape[0:2]
    img[0:50,0:width]=[0,255,0]
    
    try:

        points=landmark_detector(gray,rect[0])
        #getting the 68 points of the face
        points=face_utils.shape_to_np(points)
        #converting the points into a numpy array

        count=1
        for p in points:

            cen=(p[0],p[1])
            cv2.circle(img,cen,2,(0,255,0),-1)
            cv2.putText(img,str(count),cen,cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,255,255),1)
            count=count+1
            
        predict_face_type(points)
        
    except Exception as e:

       print(e)

    cv2.imshow('LIVE',img)
    cv2.waitKey(1)
