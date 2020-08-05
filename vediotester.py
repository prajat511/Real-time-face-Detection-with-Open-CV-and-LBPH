
import os
import cv2
#import numpy as np
import facerecogn as fr


#This module captures images via webcam and performs face recognition
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml')#Load saved training data

name = {0 : "Rajat",1 : "goutham",2:"Dodo"}


cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)

while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    faces_detected,gray_img=fr.faceDetection(test_img)


    for face in faces_detected:
        (x,y,w,h)=face
        roi_gray=gray_img[y:y+w, x:x+h]
        label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
        print("confidence:",confidence)
        print("label:",label)
        fr.draw_rect(test_img,face)
        predicted_name=name[label]
        if confidence < 70 :
            #If confidence less than 70 then  print predicted face name on screen         
            fr.put_text(test_img,predicted_name,x,y)
        else:
            name=input("print name of the person\n")
            dir = os.path.join("trainingImages",name)
            os.mkdir(dir)
            count = 0
            while True:
                ret,test_img=cap.read()
                if not ret :
                    continue
                cv2.imwrite(os.path.join(dir,"frame%d.jpg" % count), test_img)     # save frame as JPG file
                count += 1
                resized_img = cv2.resize(test_img, (1000, 700))
                cv2.imshow('face detected ',resized_img)
                if cv2.waitKey(5) == ord('q') or count==3:#wait until 'q' key is pressed
                    break
            

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('face recognition',resized_img)
    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows
