import numpy as np
import cv2


TARGET_SIZE = (960,960)
kernel = np.ones((5,5),np.uint8)

imgs=[cv2.imread('00.jpg'),cv2.imread('01.jpg'),cv2.imread('02.jpg'),cv2.imread('03.jpg'),cv2.imread('04.jpg'),cv2.imread('05.jpg'),cv2.imread('06.jpg')]

for i in range(7):
    im_resized = cv2.resize(imgs[i],TARGET_SIZE)
    rgb_planes = cv2.split(im_resized)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 19)
        diff_img =255 - cv2.absdiff(plane, bg_img)
        arr = np.array([])    
        norm_img = cv2.normalize(diff_img,arr,alpha=0.0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)

    img = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img,225,255,cv2.THRESH_BINARY)
    #equ = cv2.equalizeHist(img)
    #median = cv2.medianBlur(img, 1) 

    edges = cv2.Canny(img,50,200)

    cv2.imshow("IMG",edges) 
    cv2.waitKey()

cv2.destroyAllWindows()

#####################################################################