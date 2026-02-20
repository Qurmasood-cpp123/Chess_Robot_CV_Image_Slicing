#Image Slicing Role 1: "The Line Detective" (Masood Quraishi)


import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Storing the Chess Board Image
img =cv2.imread("board1.jpg")
resize_img=cv2.resize(img,(800,800))  # Changing The resolution

rgb_image=cv2.cvtColor(resize_img,cv2.COLOR_BGR2RGB)     
gry_image=cv2.cvtColor(resize_img,cv2.COLOR_BGR2GRAY)

plt.imshow(rgb_image) #Plotting the image on a graph
cv2.imshow("Gray_Scale",gry_image)             #Converstion of RGB to GRAY Scale

# Gaussian Blur Image Coversion
gaussian_blur=cv2.GaussianBlur(gry_image,(11,11),0)
plt.imshow(gaussian_blur,cmap="gray")        
cv2.imshow("Gaussian_Blur",gaussian_blur)         # Now Converting Gray Scale to Gaussian Blur

#OTSU Threshold (Converting the image into binary that is Black and White)
ret,otsu_binary=cv2.threshold(gaussian_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.imshow(otsu_binary,cmap="gray")
cv2.imshow("OTSU_Threshold",otsu_binary)       # Now Converting Gaussian Blur to OTSU Threshold

#Canny Edge Detection 
canny=cv2.Canny(otsu_binary,20,255);
plt.imshow(canny,cmap="gray")
cv2.imshow("Canny_Edge",canny)           # Implementing Canny Edge Dectection on the Board

#Dilation
dil=np.ones((7,7),np.uint8)
img_dilation=cv2.dilate(canny,dil,iterations=1)
plt.imshow(img_dilation,cmap="gray")
cv2.imshow("Dilation1",img_dilation)    # Dilating after Board Edge Detection


#Hough Lines
lines=cv2.HoughLinesP(img_dilation,1,np.pi/180,threshold=200,minLineLength=100,maxLineGap=50)
if lines is not None:
    for i, line in enumerate(lines):
        x1,y1,x2,y2=line[0]

        cv2.line(img_dilation,(x1,y1),(x2,y2),(255,255,255),2)

plt.imshow(img_dilation,cmap="gray")
cv2.imshow("Hough_Lines",img_dilation)    # Adding the Lines on Chess Board 

#Dilation Again after Hough Lines
dil=np.ones((3,3),np.uint8)
img_dilation_2=cv2.dilate(img_dilation,dil,iterations=1)

plt.imshow(img_dilation_2,cmap="gray")
cv2.imshow("Dilation2",img_dilation_2)  # Doing Dilation Again after adding lines on Chess Board


cv2.waitKey(0)
