import os
#used when you are dealing with the operating system

import cv2
import dlib
from imutils import face_utils





data_path='Face Shapes'
labels=os.listdir(data_path)


face_detector=dlib.get_frontal_face_detector()
#a pretrained face detecting classifier in dlib module   

landmark_detector=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#pretrained external algorithm for detecting 68 points of a face in a image

data=[]#empty arrays
target=[]

def points_68(gray):

    rects=face_detector(gray)
    #alogorithm.predict

    for rect in rects:


            points=landmark_detector(gray,rect)
            points=face_utils.shape_to_np(points)
           


            return points



def create_features(points,labels):


    target_dict={'Diamond':0,'Oblong':1,'Oval':2,'Round':3,'Square':4,'Triangle':5}



    my_points=points[2:9,0]
    #my_points contains x coordinates of p3-p9
    
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



    data.append([d1,d2,d3,d4,d5])
    #append create a new row

    target.append(target_dict[label])




    

for label in labels:


    imgs_path=os.path.join(data_path,label)#join the data path and labels
    img_names=os.listdir(imgs_path)
    print(img_names)
    print("==========================")



    for img_name in img_names:

        
        img_path=os.path.join(imgs_path,img_name)
        img=cv2.imread(img_path)
        #cv2.imshow('LIVE',img)
        #cv2.waitKey(100)

        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        points=points_68(gray)
        create_features(points,label)





import pickle
#using this library,arrays can be saved/loaded to physical files


import numpy as np

data=np.array(data)
target=np.array(target)


pickle.dump(data,open('data.pickle','wb'))
pickle.dump(target,open('target.pickle','wb'))






