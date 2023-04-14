# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 10:20:47 2023

RE4017 Project 2

roifinder.py

@author:    Finn Hourigan 17228522
            Ronan Reilly 18242367
            Brendan Lynch 18227651
            Barry Hickey 18243649
"""

from skimage.io import imread
from skimage.color import rgba2rgb
from skimage.color import rgb2hsv
from scipy import ndimage
import scipy.ndimage
import numpy as np


# define mask helepr function
# np array of image (img_arr) and thresholds array (th) in for [lower_threshold, upper_threshold] as parameters
# Returns image array after masking

def mask(img_arr, th):
    
    # Determine rows and columns of array
    r,c = img_arr.shape
    
    # Determine upper and lower threshold limits
    th_up = th[1]
    th_lo = th[0]
       
    for i in list(range(0,r)):    #loop through rows
    
        for j in list(range(0,c)):    #loop through columns
            
            #binarize to 1 or 0 depending on threshold limits
            if img_arr[i,j] <= th_up and img_arr[i,j] >= th_lo :    
                img_arr[i,j]= 0 #Any pixel WITHIN threshold limits -> 0
            else:
                img_arr[i,j] = 1 #Any pixel outside threshold limits -> 1
         
    return img_arr

# Define find_bounding_box function
# Takes a label value and a labelled array (containing a region w/ that label value) as parameters
# Returns array of the minimums and maximums for x and y in form (min_x, min_y, max_x, max_y) 
# These co-ordinates are enough to determine the bouding box of the region of interest

def find_bounding_box (label, labeled_array):
    region_mask = labeled_array == label
    indices = np.nonzero(region_mask)
    
    min_x, max_x = np.min(indices[1]), np.max(indices[1])
    min_y, max_y = np.min(indices[0]), np.max(indices[0])
    bbox = (min_x, min_y, max_x, max_y) 
    
    return bbox

    

# Define find_ROIs function
# 
# Determines regions of interest from the image that may be speed signs
#
# Takes string of image file name as argument
# Returns an array of regions of interest as numpy arrays of RGB images 

def find_ROIs(img_name):
    
    print(f"Finding regions of interest in image {img_name}...")
    
    
    
    #Read image into a 4-channel RGBA array
    img_rgba = imread(img_name)
    
    #Convert from RGBA to RGB
    img_rgb = rgba2rgb(img_rgba)
    
    #Convert from RGB to HSV
    img_hsv = rgb2hsv(img_rgb)
    
    # Split HSV into single channels   
    img_h = img_hsv[:,:,0]
    img_s = img_hsv[:,:,1]
    img_v = img_hsv[:,:,2]
    
    # Choose threshold values for H,S and V in format [lower_limit , upper_limit]
    # Any pixel WITHIN threshold limits -> 0
    # Any pixel outside threshold limits -> 1
       
    th_h = [0.025, 0.675]  
    th_s = [0, 0.3]
    th_v = [0, 0.22]
    
    # Use mask function defined above to define threshold
    
    img_h = mask(img_h,th_h)
    img_s = mask(img_s,th_s)
    img_v = mask(img_v,th_v)
      
        
    # Combine binarized channels into 1 mask
    img_mask = img_h * img_s * img_v
    
    #Max filter applied
    img_mask = scipy.ndimage.maximum_filter(img_mask,7)
    
    # #Binary erosion filter
    plus_shape= np.array([[0,1,0],[1,1,1],[0,1,0]])
    img_mask = scipy.ndimage.binary_erosion(img_mask,plus_shape)
    
    #Use ndimage.label to generate a labeled array and find the number of labels used
    labeled_array, num_features = ndimage.label(img_mask)

    # Get the size of each region and store in an array
    region_sizes = []
    for i in range(1, num_features+1):
        region_mask = labeled_array == i #region_mask refers to only areas where value is i
        region_size = np.count_nonzero(region_mask)
        region_sizes.append(region_size)
        
    
    # Find the index of the largest region
    largest_region_idx = np.argmax(region_sizes)
    
    # Compute the size threshold for removing small regions
    size_threshold = 0.3 * region_sizes[largest_region_idx]
    
    # Declare a minimum size
    # This value may be needed in the case where no speed sign is present, and therefore the ROIs determined
    # after masking are small. A 24 x 24 image (576 pixels) was decided as appropriately small for it to not in
    # fact be a discernible speed sign
    
    min_size = 576
    
    # Remove small regions and store labels of large regions (our ROIs)
    labels_to_keep = [] #To store label values of ROIs
    
    for i in range(1, num_features+1):
        if region_sizes[i-1] < size_threshold or region_sizes[i-1] < min_size:
            # Small regions below the threshold value or min size value are removed (set to 0)
            labeled_array[labeled_array == i] = 0
        else:
            # Any regions not removed have their label values added to labels_to_keep array
            labels_to_keep.append(i)
    
    roi_array = []    #For storing ROIs cut from original RGB image
    top_corner_array = [] #For storing co-ordinates of corner of ROI
    
    # Find the bounding box of each region using find_bounding_box function
    for label in (labels_to_keep):
        bbox = find_bounding_box (label, labeled_array)
       
        #Extract values from bbox array
        (min_x, min_y, max_x, max_y) = bbox
        #Length of horizontal side of box
        deltax = max_x - min_x
        #Length of vertical side of box
        deltay = max_y - min_y
        #Co-ordinates of top RH corner of box, needed for adding text to image at later stage
        top_corner = (max_x,min_y)
        
        #Remove narrow regions with a ratio greater than minratio for small side/bigger side
        minratio = 0.6
        
        # If both the ratio of height/width and width/height are above the minimum ratio,
        # the region_array is added to the array of ROI arrays
        if (deltax/deltay > minratio and deltay/deltax > minratio ):
            #Slice the RGB image to extract the subarray corresponding to the bounding box
            region_array = img_rgb[min_y:(max_y+1), min_x:(max_x+1)]
            print(f"Bounding box of region {label}: {bbox}")
            # Add ROI to storage array
            roi_array.append(region_array) 
            # Add top corner co-ordinates to array (at corresponding index to ROI)
            top_corner_array.append(top_corner)
    
    print("ROI search complete...")
    
    return roi_array, top_corner_array
          