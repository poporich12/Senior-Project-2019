import numpy as np
import cv2


TARGET_SIZE = (640,360)


img = cv2.imread('07.jpg',-1)
kernel = np.ones((7,7),np.uint8)
im_resized = cv2.resize(img,TARGET_SIZE)

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



# for plane in result_norm_planes:
#     for i in plane:
#         for j in i:
#             if(plane[i][j]>50):
#                 plane[i][j]=255
#             else:
#                  plane[i][j]=0

result = cv2.merge(result_planes)
result_norm = cv2.merge(result_norm_planes)
    


while(True):
    cv2.imshow('origin',im_resized)
    cv2.imshow('shadows_out.png', result)
    cv2.imshow('shadows_out_norm.png', result_norm)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
