import cv2
import numpy as np
from matplotlib import pyplot as plt
TARGET_SIZE = (720,720)
kernel = np.ones((5,5),np.uint8)
imgs=[cv2.imread('00.jpg',0),cv2.imread('01.jpg',0),cv2.imread('02.jpg',0),cv2.imread('03.jpg',0),cv2.imread('04.jpg',0),cv2.imread('05.jpg',0),cv2.imread('06.jpg',0)]

for i in range(7):
    #img = cv2.imread('01.jpg',0)
    _, img = cv2.threshold(imgs[i],50,255,cv2.THRESH_BINARY)
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    img = cv2.resize(imgs[i],TARGET_SIZE)
    equ = cv2.equalizeHist(img)
    new_img = cv2.convertScaleAbs(img,2,2)
    #median = cv2.medianBlur(new_img, 5) 

    #dilation = cv2.dilate(median,kernel,iterations = 15)
    #erosion = cv2.erode(median,kernel,iterations = 5)
    #opening = cv2.morphologyEx(median, cv2.MORPH_OPEN, kernel)
    #closing = cv2.morphologyEx(new_img, cv2.MORPH_CLOSE, kernel)

    #diffc = cv2.absdiff(erosion,median)
    #diffg = cv2.cvtColor(diffc,cv2.COLOR_BGR2GRAY)
    #bwmask = cv2.inRange(diffg,50,255)

    edges = cv2.Canny(img,100,150)

    cv2.imshow("IMG",new_img) 
    cv2.waitKey()

cv2.destroyAllWindows()