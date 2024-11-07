import os  # Imports the os module for interacting with the file system
import cv2 as cv  # Imports OpenCV for image processing
import numpy as np  # Imports NumPy for numerical operations and handling arrays

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']  # List of names representing people in the dataset
DIR = r'D:\opencv\faces\train'  # Directory path where the training images are stored

haar_cascade = cv.CascadeClassifier(r'D:\opencv\code\haar_face.xml')  # Loads the Haar Cascade classifier for face detection

features = []  # List to store facial regions (ROIs) detected from images
labels = []  # List to store labels corresponding to each face ROI

def create_train():  # Defines a function to process images and extract faces for training
    for person in people:  # Loops through each person in the list
        path = os.path.join(DIR, person)  # Constructs the path to the folder for each person
        label = people.index(person)  # Gets the index of the person as a label

        for img in os.listdir(path):  # Loops through each image file in the person's folder
            img_path = os.path.join(path, img)  # Constructs the full path to the image file

            img_array = cv.imread(img_path)  # Reads the image file into an array
            if img_array is None:  # Checks if the image was read correctly
                continue  # Skips to the next image if the current image could not be read

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)  # Converts the image to grayscale for better face detection

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)  # Detects faces in the grayscale image

            for (x, y, w, h) in faces_rect:  # Loops over each detected face
                faces_roi = gray[y:y+h, x:x+w]  # Extracts the region of interest (ROI) of the face
                features.append(faces_roi)  # Adds the face ROI to the features list
                labels.append(label)  # Adds the corresponding label to the labels list

create_train()  # Calls the function to populate features and labels with data
print('Training done ---------------')  # Prints a message indicating completion of data collection

features = np.array(features, dtype='object')  # Converts the features list to a NumPy array for training
labels = np.array(labels)  # Converts the labels list to a NumPy array

face_recognizer = cv.face.LBPHFaceRecognizer_create()  # Initializes an LBPH face recognizer for face recognition

face_recognizer.train(features, labels)  # Trains the recognizer on the features and labels

face_recognizer.save('face_trained.yml')  # Saves the trained model to a file
np.save('features.npy', features)  # Saves the features array to a file
np.save('labels.npy', labels)  # Saves the labels array to a file
