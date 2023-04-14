# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 18:52:30 2023

RE4017 Project 2

signdetector.py

NB Command line argument of img file name including extension required

@author:    Finn Hourigan 17228522
            Ronan Reilly 18242367
            Brendan Lynch 18227651
            Barry Hickey 18243649

"""
import matplotlib.pyplot as plt
import roifinder
import classifier
import sys

from PIL import Image, ImageFont, ImageDraw


# Command line argument is taken as name of image file to be processed
img_name = sys.argv[1]
 
   
#Find all ROIs and their corresponding top RH corner co-ordinates
roi_array, top_corner_array = roifinder.find_ROIs(img_name)


# Speed limit results storage array initialised
speed_limits_arr = []

#Find speed limit associated with each roi using classifier, add to storage array
for roi in roi_array:
    speed_limit = classifier.classify_img(roi)
    speed_limits_arr.append(speed_limit) 
   
# Translation operation, co-ordinates of text location defined
text_loc = [(x - 100, y - 60) for x, y in top_corner_array]

# Translation operation, co-ordinates of top LH corner of label box defined
box_corner1 = [(x - 20, y - 20) for x, y in text_loc]
# Translation operation, co-ordinates of bottom RH corner of label box defined
box_corner2 = [(x + 230 , y + 80) for x, y in box_corner1 ]

# Load the original image
img = Image.open(img_name)
width, height = img.size

# Create a new image (blank, same dimensions)
img_with_text = Image.new('RGB', (width, height), color=(255, 255, 255))

# Copy original image onto the img_with_text
img_with_text.paste(img, (0, 0))

# Create new ImageDraw object
draw = ImageDraw.Draw(img_with_text)
font = ImageFont.truetype("arial.ttf", 50)

# Counter for no. of actual speed signs present,
# as some ROIs are determined to not be a speed sign 
sign_counter = 0

for i in list(range(len(roi_array))):
    # Image vectors used for "hard-negative mining" return a speed limit of -1
    # The following if-loop prevents them from producing a text label
    if speed_limits_arr[i] > 0:
        sign_counter+=1
        # Co-ordinates of corners of labels used to draw white rectangle
        draw.rectangle([box_corner1[i],box_corner2[i] ] , fill="white")
        # Text denoting speed limit added to image            
        draw.text(text_loc[i], f"{int(speed_limits_arr[i])} km/h", font=font, fill=(255, 165, 0))
        print(f"Speed sign {sign_counter}: {int(speed_limits_arr[i])} km/h")

# If no speed signs were detected, print message to terminal
# This could be due to no signs picked up by the roifinder.find_ROIs function, or all signs picked up were
# removed by the "hard-negative mining" image vectors
if sign_counter == 0:
    print("No signs detected!") 
    
# Show final image        
plt.imshow(img_with_text)
plt.title(img_name)
plt.axis('off')
plt.show()
    
    

        
    
        