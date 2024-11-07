import cv2 as cv
import numpy as np

image = cv.imread(r'D:\opencv\photos\park.jpg')
cv.imshow('park', image)

# transelation

# -x => left , x => right || -y => up , y => down

def transelated(image,x,y) :
    trans_matrix = np.float32([[1,0,x],[0,1,y]])
    dimension = (image.shape[1],image.shape[0])
    return cv.warpAffine(image,trans_matrix,dimension)

transelation = transelated(image,50,-25)

cv.imshow('transelation',transelation)


# Rotation
def rotate(image, angle, rotPoint=None):
    (height,width) = image.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2,height//2)
    
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width,height)

    return cv.warpAffine(image, rotMat, dimensions)

rotation = rotate(image, 30)
cv.imshow('rotation', rotation)




cv.waitKey(0)