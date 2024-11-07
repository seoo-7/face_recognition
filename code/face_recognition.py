import numpy as np
import cv2 as cv

# Initialize Haar Cascade and recognizer
haar_cascade = cv.CascadeClassifier(r'D:\opencv\code\haar_face.xml')
face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.read(r'D:\opencv\code\face_trained.yml')  

# List of people (class labels)
people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

# Load an image for testing
img = cv.imread(r'D:\opencv\faces\test\2.jpg')  
if img is None:
    raise FileNotFoundError("Image file not found.")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detect faces in the image
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]

    # Perform prediction
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label => {people[label]} with a confidence of {confidence}%')

    # Display label and confidence on the image
    label_text = f"{people[label]}: {confidence:.2f}"
    cv.putText(img, label_text, (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), thickness=2)
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

cv.imshow('Detected Face', img)
cv.waitKey(0)
cv.destroyAllWindows()
