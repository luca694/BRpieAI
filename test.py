import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained face recognition model
model = load_model("model001.h5")


labels = ["Kb","kin","ryk"]

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

        
        face_roi = grey[y:y+h, x:x+w]

        
        face_resized = cv2.resize(face_roi, (100, 100))
        face_resized = face_resized / 255.0  # normalize
        face_resized = np.expand_dims(face_resized, axis=-1)
        face_resized = np.expand_dims(face_resized, axis=0)

        
        predictions = model.predict(face_resized)
        class_id = np.argmax(predictions)
        confidence = np.max(predictions)

        # Get label
        name = labels[class_id]

        # Put text above face
        cv2.putText(res, f"{name} ({confidence:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', res)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
