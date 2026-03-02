#Image Slicing Role 1: "The Line Detective" (Masood Quraishi)


import cv2
import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd


#Storing the Chess Board Image
img =cv2.imread("board1.jpg")
resize_img=cv2.resize(img,(860,860))  # Changing The resolution

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




# Role 2: Neve Turner: Find and Filter Contours

#Find all the contours from the dilated image
board_contours, hierarchy = cv2.findContours(img_dilation_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# List will store center coordinates and corner points of detected squares
square_centers = list()

board_squared = canny.copy()

# Iterate through all detected contours
for contour in board_contours:
    if 4000 < cv2.contourArea(contour) < 20000:
        epsilon = 0.02 * cv2.arcLength(contour, True) 
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Keeping only square contours 
        if len(approx) == 4:
            pts = [pt[0] for pt in approx] # Extracting coordinates

            # Defining the points
            pt1 = tuple(pts[0])
            pt2 = tuple(pts[1])
            pt3 = tuple(pts[2])
            pt4 = tuple(pts[3])

            x, y, w, h = cv2.boundingRect(contour)

            # Calculating the center of the square
            center_x= x+w/2
            center_y= y+h/2

            # Storing square information in list
            square_centers.append([center_x, center_y, pt2, pt1, pt3, pt4])
            
            # Drawring the edges of the detected square 
            cv2.drawContours(board_squared, [approx], -1, (255, 255, 0), 7)

plt.imshow(board_squared, cmap="grey")

cv2.imshow("Dectected squares", board_squared)


# Sorting coordinates
sorted_coordinates = sorted(square_centers, key=lambda x: x[1], reverse=True)

groups = []
current_group = [sorted_coordinates[0]]

for coord in sorted_coordinates[1:]:
    if abs(coord[1] - current_group[-1][1]) < 50:
        current_group.append(coord)
    else:
        groups.append(current_group)
        current_group = [coord]

groups.append(current_group) # Append the last group


for group in groups: # Sort each group by the second index (column values)
    group.sort(key=lambda x: x[0])


sorted_coordinates = [coord for group in groups for coord in group] # Combine the groups back together

# Finding undetected squares
for num in range(len(sorted_coordinates)-1): 
    if abs(sorted_coordinates[num][1] - sorted_coordinates[num+1][1])< 50 :
        if sorted_coordinates[num+1][0] - sorted_coordinates[num][0] > 150:
            x=(sorted_coordinates[num+1][0] + sorted_coordinates[num][0])/2
            y=(sorted_coordinates[num+1][1] + sorted_coordinates[num][1])/2
            p1=sorted_coordinates[num+1][5]
            p2=sorted_coordinates[num+1][4]
            p3=sorted_coordinates[num][3]
            p4=sorted_coordinates[num][2]
            sorted_coordinates.insert(num+1,[x,y,p1,p2,p3,p4])

num_squares = len(sorted_coordinates)
print(f"Total squares detected: {num_squares}")

# Group into rows of 8 dynamically
for i in range(0, num_squares, 8):
    row = sorted_coordinates[i : i + 8]
    # This prints the (x, y) center for each square in the row
    print(f"Row {i//8}: {[[round(c[0]), round(c[1])] for c in row]}")

board_vis = cv2.cvtColor(board_squared, cv2.COLOR_GRAY2BGR)
square_num=1
for cor in sorted_coordinates:
  cv2.putText(img = board_vis,text = str(square_num),org = (int(cor[0])-30, int(cor[1])),
    fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale = 1,color = (255, 246, 55),thickness = 1)
  square_num+=1


cv2.imshow("Numbered squares", board_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()