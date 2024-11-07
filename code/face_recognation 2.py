import numpy as np  # Imports NumPy for handling arrays and numerical operations
import cv2 as cv  # Imports OpenCV for image processing and computer vision tasks

# Initialize Haar Cascade and recognizer
haar_cascade = cv.CascadeClassifier(r'D:\opencv\code\haar_face.xml')  # Loads the Haar Cascade classifier for detecting faces
face_recognizer = cv.face.LBPHFaceRecognizer_create()  # Initializes an LBPH face recognizer for face recognition

face_recognizer.read(r'D:\opencv\code\face_trained.yml')  # Loads a pre-trained face recognizer model from a file

# List of people (class labels)
people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']  # Defines names for class labels

# Load an image for testing
img = cv.imread(r'D:\opencv\faces\val\elton john\3.jpg')  # Reads the test image into an array
if img is None:  # Checks if the image file was loaded successfully
    raise FileNotFoundError("Image file not found.")  # Raises an error if the file is not found

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Converts the test image to grayscale for better detection performance
cv.imshow('Person', gray)  # Displays the grayscale image in a window named 'Person'

# Detect faces in the image
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)  # Detects faces in the grayscale image

for (x, y, w, h) in faces_rect:  # Iterates over each detected face
    faces_roi = gray[y:y+h, x:x+w]  # Extracts the region of interest (ROI) for each detected face

    # Perform prediction
    label, confidence = face_recognizer.predict(faces_roi)  # Predicts the label and confidence for each detected face
    print(f'Label => {people[label]} with a confidence of {confidence}%')  # Prints the predicted label and confidence

    # Display label and confidence on the image
    label_text = f"{people[label]}: {confidence:.2f}"  # Prepares the label text with the name and confidence level
    cv.putText(img, label_text, (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), thickness=2)  # Draws the label text on the image
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)  # Draws a rectangle around each detected face

cv.imshow('Detected Face', img)  # Displays the final image with detected faces and labels
cv.waitKey(0)  # Waits for a key press to close the displayed windows
cv.destroyAllWindows()  # Closes all OpenCV windows
