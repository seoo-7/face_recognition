import cv2 as cv

image = cv.imread(r'D:\opencv\photos\park.jpg')
cv.imshow('park',image)

# # convert the photo to grayscale
# gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) 
# cv.imshow('gray',gray)

# # convert the image to blur
# blur = cv.GaussianBlur(image , (7,7),cv.BORDER_DEFAULT)
# cv.imshow('blur', blur)

# # convert the image to canny edges
# canny = cv.Canny(image, 100 , 120)
# cv.imshow('canny edge',canny)

# resize the image
resaize = cv.resize(image,(500,500), interpolation= cv.INTER_CUBIC)
cv.imshow('resize',resaize) 



cv.waitKey(0)
cv.destroyAllWindows()