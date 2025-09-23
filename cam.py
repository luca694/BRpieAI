import cv2
import numpy as np

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    
    ret, frame = cap.read()
    if not ret:
        break
    
    
    res = cv2.resize(frame, (640, 480))  
    grey = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    
    
    faces = face_cascade.detectMultiScale(
        grey,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    
    for (x, y, w, h) in faces:
        cv2.rectangle(res, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    data = []
        
    data.append(grey)
    
    cv2.imshow('Face Detection', res)
    
    #note here form yourself. press a fucking q to quit in case you are no smart enough to figure it out
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
