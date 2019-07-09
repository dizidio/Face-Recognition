import cv2
import os

cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('./Classifiers/haarcascade_frontalface_default.xml')
i=0
#offset=50
name='1'

while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        i=i+1
        cv2.imwrite("dataSet/face-"+name +'.'+ str(i) + ".jpg", gray[y:y+h,x:x+w])
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        cv2.imshow('im',im[y:y+h,x:x+w])
        cv2.waitKey(100)
    if i>200:
        cam.release()
        cv2.destroyAllWindows()
        break

