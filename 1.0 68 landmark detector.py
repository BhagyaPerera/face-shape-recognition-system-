import cv2
import dlib

camera=cv2.VideoCapture(0)#here we can give the vedio or cctv cameras with the ip address,username and pastword
#camera is a video object,0-default camera,1,2-webcams/usb cams connected
#file_path can also be given,ip adress for wifi cameras


while(True) :

    ret,img=camera.read()
    #capturing 1 frame from the vedio source in 'camera object' and saves it in 'img'
    #ret is boolean value,1-camera is available,0-camera is not available




    if(ret==True):
        img[50:100,50:100]=[255,0,0]

        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #img-source image,cv2.COLOR-BGR2GRAY-color conversion flag
        #gray-resultant gray scaled image

        
        cv2.imshow('IMG',img)   
        cv2.imshow('GRAY',gray)
        
        cv2.waitKey(1)
        #1ms delay inbetween each frames
