import cv2 as cv  # type: ignore

# Load an image from a file
image = cv.imread(r'D:\opencv\photos\lady.jpg')

# Check if the image was loaded successfully
if image is None:
    print("Error: Cannot open image.") 
    exit()

# Display the image in a window
cv.imshow('Image', image)

# Wait for a key press indefinitely or for a specified amount of time in milliseconds
cv.waitKey(0)

# Destroy all the windows when the key is pressed
cv.destroyAllWindows()
