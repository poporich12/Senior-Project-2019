'''
Copyright (c) 2017 Intel Corporation.
Licensed under the MIT license. See LICENSE file in the project root for full license information.
'''

import cv2
import sys
import numpy as np
from numpy import arctan
from scipy import ndimage
import pymysql as m
#import paho.mqtt.client as mqtt
import datetime
import os

from flask import Flask, request, Response
import jsonpickle
import cv2
import numpy
import werkzeug
app = Flask(__name__)


def avg_circles(circles, b):
    avg_x = 0
    avg_y = 0
    avg_r = 0
    for i in range(b):
        # optional - average for multiple circles (can happen when a gauge is at a slight angle)
        avg_x = avg_x + circles[0][i][0]
        avg_y = avg_y + circles[0][i][1]
        avg_r = avg_r + circles[0][i][2]
    avg_x = int(avg_x/(b))
    avg_y = int(avg_y/(b))
    avg_r = int(avg_r/(b)*0.75)
    return avg_x, avg_y, avg_r


def dist_2_pts(x1, y1, x2, y2):    
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def find_angleLine(x, y, r, img, lower, upper):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lineleft = cv2.inRange(hsv, lower, upper)  

    # cv2.imshow("line",lineleft)
    # cv2.waitKey()
    # cv2.destroyWindow("line")

    minLineLength = 5
    lines1 = cv2.HoughLinesP(image=lineleft, rho=3, theta=np.pi/180, threshold=40, minLineLength=minLineLength)

    linesInRange=[]
    for i in range(0, len(lines1)):            
        for x1, y1, x2, y2 in lines1[i]:
            if dist_2_pts(x,y,x1,y1)<=r and dist_2_pts(x,y,x2,y2)<=r:
                linesInRange.append([x1,y1,x2,y2])

    max_dis = 0
    for i in range(0, len(linesInRange)):
        for x1_c, y1_c, x2_c, y2_c in lines1[i]:
            if max_dis < dist_2_pts(x1_c, y1_c, x2_c, y2_c):
                max_dis = dist_2_pts(x1_c, y1_c, x2_c, y2_c)
                x1, x2, y1, y2 = x1_c, x2_c, y1_c, y2_c

    return find_angleLine2(x, y, x1, y1, x2, y2, img)


def find_angleLine2(x, y, x1, y1, x2, y2, img):

    # find the farthest point from the center to be what is used to determine the angle
    img_copy = img.copy()

    dist_pt_0 = dist_2_pts(x, y, x1, y1)
    dist_pt_1 = dist_2_pts(x, y, x2, y2)
    if (dist_pt_0 > dist_pt_1):
        x_angle = x1 - x
        y_angle = y - y1
        cv2.circle(img_copy, (x1, y1), 20, (0, 0, 255), 3, cv2.LINE_AA)
        # img[x1,y1]=[0,0,255]
        # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        x_angle = x2 - x
        y_angle = y - y2
        cv2.circle(img_copy, (x2, y2), 20, (0, 0, 255), 3, cv2.LINE_AA)
        # img[x2,y2]=[0,0,255]
        # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if x_angle == 0:
        return 0
    else:
        res = np.arctan(np.divide(float(y_angle), float(x_angle)))
        # np.rad2deg(res) #coverts to degrees

        #these were determined by trial and error
        res = np.rad2deg(res)
        if x_angle > 0 and y_angle > 0:  #in quadrant I
            final_angle = 270 - res
        if x_angle < 0 and y_angle > 0:  #in quadrant II
            final_angle = 90 - res
        if x_angle < 0 and y_angle < 0:  #in quadrant III
            final_angle = 90 - res
        if x_angle > 0 and y_angle < 0:  #in quadrant IV
            final_angle = 270 - res

        return final_angle
def rgb2hsv(rgb):
    percent=15
    lower=[]
    upper=[]

    r=int(rgb[1:3],16)
    g=int(rgb[3:5],16)
    b=int(rgb[5:7],16)

    rgb_arr=np.uint8([[[b,g,r]]])
    hsv = cv2.cvtColor(rgb_arr,cv2.COLOR_BGR2HSV)

    for i in range(3):        
        lower.append(hsv[0][0][i]-(hsv[0][0][i]*percent/100))
        if i == 0:
            upper.append(hsv[0][0][i]+(hsv[0][0][i]*percent/100))
        else:
            upper.append(255)

    return lower,upper,[b,g,r]

def calibrate_gauge(gauge_name,color1,color3):
    '''
        This function should be run using a test image in order to calibrate the range available to the dial as well as the
        units.  It works by first finding the center point and radius of the gauge.  Then it draws lines at hard coded intervals
        (separation) in degrees.  It then prompts the user to enter position in degrees of the lowest possible value of the gauge,
        as well as the starting value (which is probably zero in most cases but it won't assume that).  It will then ask for the
        position in degrees of the largest possible value of the gauge. Finally, it will ask for the units.  This assumes that
        the gauge is linear (as most probably are).
        It will return the min value with angle in degrees (as a tuple), the max value with angle in degrees (as a tuple),
        and the units (as a string).
    '''
    img = cv2.imread('C:/xampp7/htdocs/myproject/%s' % (gauge_name))
    
    try:
     height, width = img.shape[:2]
    except AttributeError:
     print("shape not found")
     print("number gauge ",gauge_name)
   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #convert to gray
    #for testing, output gray image
    #cv2.imwrite('gauge-%s-bw.%s' %(gauge_number, file_type),gray)

    #detect circles
    #restricting the search from 35-48% of the possible radii gives fairly good results across different samples.  Remember that
    #these are pixel values which correspond to the possible radii search range.
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, np.array([]), 10, 5, int(height*0.35), int(height*0.48))
    # average found circles, found it to be more accurate than trying to tune HoughCircles parameters to get just the right one
    try:
        a, b, c = circles.shape
    except(ValueError):
        print("gauge ",gauge_name," has problem")

    lower1,upper1,_=rgb2hsv(color1)
    lower2,upper2,_=rgb2hsv(color3)   

    x,y,r = avg_circles(circles, b)    
    #get user input on min, max, values, and units   
    min_angle = find_angleLine(x,y,r,img.copy(),np.array(lower1),np.array(upper1))
    max_angle = find_angleLine(x,y,r,img.copy(),np.array(lower2),np.array(upper2))    
 
    
    return min_angle, max_angle, x, y, r


def ilumination(img):

    dilated_img = cv2.dilate(img, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 19)
    diff_img = 255 - cv2.absdiff(img, bg_img)

    return diff_img


def get_current_value(img, min_angle, max_angle, min_value, max_value, x, y, r, gauge_name,arrow_color):   
    
    lower,upper,rgb_arr=rgb2hsv(arrow_color)
    if(rgb_arr[0]<=64 and rgb_arr[1]<=64 and rgb_arr[2]<=64 ):

        img = ilumination(img)
        gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Set threshold and maxValue
        thresh = 200
        maxValue = 255

        # apply thresholding which helps for finding lines
        th, dst2 = cv2.threshold(gray2, thresh, maxValue, cv2.THRESH_BINARY_INV);
    
    else:
        kernel = np.ones((5,5),np.uint8)
        dst= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        dst2=cv2.inRange(dst,np.array(lower),np.array(upper))
        dst2 = cv2.dilate(dst2,kernel,iterations = 1)


    
    cv2.imshow("threshold",dst2)
    cv2.waitKey()
    cv2.destroyWindow("threshold")
   
    # find lines
    minLineLength = 70    
    lines = cv2.HoughLinesP(image=dst2, rho=3, theta=np.pi / 180, threshold=100,minLineLength=minLineLength,maxLineGap=0)  # rho is set to 3 to detect more lines, easier to get more then filter them out later

    

    # remove all lines outside a given radius
    final_line_list = []
    #print "radius: %s" %r

    diff1LowerBound = 0.0 #diff1LowerBound and diff1UpperBound determine how close the line should be from the center
    diff1UpperBound = 0.3
    diff2LowerBound = 0.45 #diff2LowerBound and diff2UpperBound determine how close the other point of the line should be to the outside of the gauge
    diff2UpperBound = 1   

    for i in range(0, len(lines)):            
        for x1, y1, x2, y2 in lines[i]:
            diff1 = dist_2_pts(x, y, x1, y1)  # x, y is center of circle
            diff2 = dist_2_pts(x, y, x2, y2)  # x, y is center of circle
            #set diff1 to be the smaller (closest to the center) of the two), makes the math easier
            if (diff1 > diff2):
                temp = diff1
                diff1 = diff2
                diff2 = temp
                # check if line is within an acceptable range           
            if (((diff1<=diff1UpperBound*r) and (diff1>=diff1LowerBound*r) and (diff2<=diff2UpperBound*r)) and (diff2>=diff2LowerBound*r)):                
                line_length = dist_2_pts(x1, y1, x2, y2)
                # add to final list
                final_line_list.append([x1, y1, x2, y2])    
    
    img_copy=img.copy()
    # cv2.circle(img_copy, (x, y), r, (0, 0, 255), 3, cv2.LINE_AA)  # draw circle
    # cv2.imshow("circle",img_copy)
    # cv2.waitKey()
    # cv2.destroyWindow("circle")


    #testing only, show all lines after filtering
    if(len(final_line_list)!=0):

        x1 = final_line_list[int(len(final_line_list)/2)][0]
        y1 = final_line_list[int(len(final_line_list)/2)][1]
        x2 = final_line_list[int(len(final_line_list)/2)][2]
        y2 = final_line_list[int(len(final_line_list)/2)][3]       
        cv2.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 3, cv2.LINE_AA)
    

        # cv2.circle(img_copy, (x, y), r, (0, 0, 255), 3, cv2.LINE_AA)  # draw circle     
        cv2.imshow("arrow",img_copy)
        cv2.waitKey()
        cv2.destroyWindow("arrow")

        final_angle=find_angleLine2(x,y,x1,y1,x2,y2,img.copy())
        
        # print("arrow_angle",final_angle)     
    
        old_min = float(min_angle)
        old_max = float(max_angle)

        new_min = float(min_value)
        new_max = float(max_value)
        # print(new_min)
        # print(new_max)

        old_value = final_angle

        old_range = (old_max - old_min)
        new_range = (new_max - new_min)
        new_value = (((old_value - old_min) * new_range) / old_range) + new_min

        return new_value
    else:
        return Exception




def rotate(gauge_name,color2):
    img = cv2.imread('C:/xampp7/htdocs/myproject/%s' % (gauge_name))
    scale_percent = 15
    try:
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        
        img = resized[25:height-25,:width]        
    except AttributeError:
        print("shape not found")
        print("number gauge ",gauge_name)
    
    # img = cv2.resize(img,(shape,shape))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #convert to gray    
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10, np.array([]), 10, 5,int(height*0.35),int(height*0.48))
    # average found circles, found it to be more accurate than trying to tune HoughCircles parameters to get just the right one
    try:
        a, b, c = circles.shape
    except(ValueError):
        print("gauge has problem")        

    x,y,r = avg_circles(circles, b)

    lower,upper,_=rgb2hsv(color2)

    degree = find_angleLine(x,y,r,img,np.array(lower),np.array(upper))  
    rotated = ndimage.rotate(img, degree)

    cv2.imwrite('C:/xampp7/htdocs/myproject/mo_%s' % (gauge_name), rotated) 

   


def main_1(filename, min_value, max_value, unit,color1,color2,color3,arrow_color):   
    print("filename "+filename)
    # img = cv2.imread('C:/xampp7/htdocs/myproject/%s' % (filename))
    # img = ndimage.rotate(img, -90)
    # cv2.imwrite('C:/xampp7/htdocs/myproject/%s' % (filename),img)

    print("state1")
    rotate(filename,color2)  
    print("state2")  
    min_angle, max_angle, x, y, r = calibrate_gauge("mo_"+filename,color1,color3)
    img = cv2.imread('C:/xampp7/htdocs/myproject/mo_%s' % (filename))
   
    # img = cv2.resize(img,(shape,shape))
    print("state3")
    val = get_current_value(img, min_angle, max_angle,
                            min_value, max_value, x, y, r, filename,arrow_color)
    
    return val
    # c = None
    # try:
    #     c = m.connect(host='localhost', user='root', passwd='', db='mydb')
    #     cur = c.cursor()
    #     cur.execute('SET NAMES utf8;')
    #     loop = 1;
    #     time = datetime.datetime.now()
    #     company_id = 1
    #     thermometer_id = 1
    #     thermo_type_id = 1

    #     while loop==1:
    #         if time=='exit' or company_id=='exit' or thermometer_id=='exit' or thermo_type_id=='exit' :
    #             loop = 0
    #             continue

    #     sql = "INSERT INTO `my` (`val`, `time` , `company_id`, `thermometer_id`,`thermo_type_id`) \
    #                     VALUE (NULL, '%s ','%s', '%s') " \
    #                     %(val,time,company_id,thermometer_id,thermo_type_id)
    #     sql = sql.encode('utf-8')
    #     try:
    #         cur.execute(sql)
    #         c.commit()
    #         print('เพิ่มข้อมูล เรียบร้อยแล้ว')
    #     except:
    #         c.rollback()
    #         print('เพิ่มข้อมูล ผิดพลาด')
    # except m.Error:
    #     print('ติดต่อฐานข้อมูลผิดพลาด')
    # if c:
    #     c.close()
data={
    "Path":"None",
    "Type":"Nono",
    "Min Value":"None",
    "Max Value":"None",
    "Unit":"None",
    "RGBColor1":"None",
    "RGBColor2":"None",
    "RGBColor3":"None",
    "RGBArrowColor":"None"}

@app.route('/data2analog',methods=['GET','POST'])
def handle_data():
    if request.method == 'POST':        
        data = request.json


    val = str(main_1(data["Path"],data["Min Value"],data["Max Value"],data["Unit"],
                    data["RGBColor1"],data["RGBColor2"],data["RGBColor3"],data["RGBArrowColor"]))
    
    print("Current reading: %s %s" %(val, data["Unit"]))



    return data

@app.route('/img', methods=['GET', 'POST'])
def handle_request():   
    imagefile = request.files['image']
    
    if request.method == 'POST':
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        print("\nReceived image File name : " + imagefile.filename)
        imagefile.save(filename)
        
    return "Uploaded Success!!!"


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
