import cv2 as cv

# Load the image
img = cv.imread(r'D:\opencv\photos\3.webp')
if img is None:
    print("Error: Image not found.")
    exit()

# Display the original image
cv.imshow('that is me', img)

# Convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray People', gray)

# Load the Haar Cascade file
haar_cascade = cv.CascadeClassifier(r'D:\opencv\code\haar_face.xml')
if haar_cascade.empty():
    print("Error: Haar cascade file not loaded.")
    exit()

# Detect faces
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
print(f'Number of faces found = {len(faces_rect)}')

# Draw rectangles around detected faces
for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

# Show the result
cv.imshow('Detected Faces', img)

cv.waitKey(0)
cv.destroyAllWindows()
