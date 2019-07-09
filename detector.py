import cv2,os
import numpy as np
from PIL import Image 

#path = os.path.dirname(os.path.abspath(__file__))

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('./trainer/trainer.yml')
cascadePath = "./Classifiers/haarcascade_frontalface_default.xml"
cascadePath2 = "./Classifiers/haarcascade_profileface.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
faceCascade2 = cv2.CascadeClassifier(cascadePath2);

cam = cv2.VideoCapture(0)
#font = cv2.CV_FONT_HERSHEY_SIMPLEX #Creates a font
#filename = 'busy.jpg'
#im = cv2.imread(filename)
c=0
while c != 27:
    ret, im = cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5,
                                       minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE)
    faces2=faceCascade2.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5,
                                       minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE)
    faces = np.asarray(faces, dtype='int32');
    faces2 = np.asarray(faces2, dtype='int32');
    faces = faces if faces.size else faces2;

    for(x,y,w,h) in faces:
        nbr_predicted, conf = recognizer.predict(gray[y:y+h,x:x+w])
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        if(nbr_predicted==0):
             nbr_predicted='Diogo'
        elif(nbr_predicted==1):
             nbr_predicted='Pedro'
        cv2.putText(im,str(nbr_predicted)+"--"+str(conf), (x,y+h),cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0)) #Draw the text
    cv2.imshow('im',im)
    c = cv2.waitKey(1)
            
#cv2.imwrite('./output/{}'.format(filename), im);
#cv2.waitKey(0)
cam.release()
cv2.destroyAllWindows()







