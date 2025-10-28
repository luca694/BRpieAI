import os
import cv2
import numpy as np

data_dir = r"C:\Users\khoo4\whatthistime\db"

X = []  
y = []  

for label, person in enumerate(os.listdir(data_dir)):
    person_folder = os.path.join(data_dir, person)
    if os.path.isdir(person_folder):
        for file in os.listdir(person_folder):
            img_path = os.path.join(person_folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
            img = cv2.resize(img, (100, 100))  
            X.append(img)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Dataset shape:", X.shape)
print("Labels:", np.unique(y))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)



