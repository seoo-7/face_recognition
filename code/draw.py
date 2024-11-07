import cv2 as cv
import numpy as np

#Create a blank image (black) with a specific size and 3 color channels (RGB)
#The size is (height, width, number of channels)
blank_image = np.zeros((500, 500, 3), dtype='uint8')  # 500x500 pixels, 3 channels (BGR), black image
# Display the blank image
cv.imshow('Blank Image', blank_image)

'''
# to paint  square you do
blank_image[200:300 , 300:400] = 0,0,255
cv.imshow('square', blank_image) '''

# draw the rectangle
cv.rectangle(blank_image, (0,0), (250,250),(0,0,250 ),thickness=-1)
cv.imshow('rectangle', blank_image)

#draw the circle
cv.circle(blank_image,(blank_image.shape[0]//2,blank_image.shape[1]//2), 75, (0,120,0),thickness=3)
cv.imshow('circle', blank_image)

# draw line
cv.line(blank_image, (100,250), (300,400), (255,255,255), thickness=3)
cv.imshow('line', blank_image)

# write the text 
cv.putText(blank_image,'  hi my name is seoo  ',(0,225),cv.FONT_HERSHEY_SCRIPT_COMPLEX,1.0,(0,250,0),2)
cv.imshow('line', blank_image)

# Wait indefinitely until a key is pressed
cv.waitKey(0)

# Destroy all OpenCV windows
cv.destroyAllWindows()

