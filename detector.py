import cv2,os
import numpy as np
from PIL import Image 

#path = os.path.dirname(os.path.abspath(__file__))

#recognizer = cv2.createLBPHFaceRecognizer()
#recognizer.load(path+r'\trainer\trainer.yml')
cascadePath = "./Classifiers/haarcascade_frontalface_default.xml"
cascadePath2 = "./Classifiers/haarcascade_profileface.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
faceCascade2 = cv2.CascadeClassifier(cascadePath2);
#cam = cv2.VideoCapture(0)
#font = cv2.CV_FONT_HERSHEY_SIMPLEX #Creates a font
filename = 'busy.jpg'
im = cv2.imread(filename)
gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5,
                                   minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE)
faces2=faceCascade2.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5,
                                   minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE)
for(x,y,w,h) in np.concatenate((faces,faces2)):
    #nbr_predicted, conf = recognzier.predict(gray[y:y+h,x:x+w])
    cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
    #if(nbr_predicted==7):
    #     nbr_predicted='Obama'
    #elif(nbr_predicted==2):
    #     nbr_predicted='Anirban'
    #cv2.PutText(im,str(nbr_predicted)+"--"+str(conf), (x,y+h),font, 1.1, (0,255,0)) #Draw the text
#cv2.imshow('im',gray)

cv2.imwrite('./output/{}'.format(filename), im);
#cv2.waitKey(0)

#cv2.destroyAllWindows()







