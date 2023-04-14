# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 21:13:50 2023

RE4017 Project 2

classifier.py

@author:    Finn Hourigan 17228522
            Ronan Reilly 18242367
            Brendan Lynch 18227651
            Barry Hickey 18243649
"""


from skimage.transform import resize
from skimage.color import rgb2gray

import numpy as np

# Function for determing the euclidian distnace between two vectors of same dimension
# Takes two vectors as arguments
# Return distance between
def distance_between_vectors (vector1, vector2):
     
    # finding sum of squares
    sum_sq = np.sum(np.square(vector1 - vector2))
     
    # Doing squareroot
    dist = np.sqrt(sum_sq)
    
    return dist


# Function for classifying a region of interest (roi) as a given speed sign value
# Determines the speed sign value according to our 1-NN classifier and the given descriptor vectors .npy file 
#
# Takes a numpy array representing RGB data of the roi as an argument
# Returns the speed sign value determined

def classify_img(roi):
       
    # convert to grayscale
    gray_img = rgb2gray(roi)
    
    
    # Resize the image to 64x64 pixels
    
    resized_img = resize(gray_img, (64, 64), anti_aliasing=True)
    
    #Determine max and min grayscale values of array
    max_g = resized_img.max()
    min_g = resized_img.min()
    
    #Rescale to int values 0 - 255 and contrast enhance by using full 0 - 255 scale
    resized_img = (resized_img - min_g) * (255/(max_g - min_g))
    
    
    # Find mean pixel value of resized image
    mean_pixel_value = np.mean(resized_img)
    
    # Subtract the mean pixel value from all elements of resized image to find the "zero mean image"
    zero_mean_img = resized_img - mean_pixel_value
    
    # Create a 4096-elemnt vector from the 64 x 64 "zero mean image" using .fLatten()
    img_vector = zero_mean_img.flatten()
    
    # Find the value of the norm for this vector
    norm =np.linalg.norm(img_vector)
    
    # Normalise vector by dividing by the norm value
    # This 4096 element vector is to be compared to the descriptor vectors supplied
    norm_img_vector = img_vector/norm
    
    
    #Load descriptor vectors .npy file
    mat = np.load("1-NN-descriptor-vects.npy")
    
    #Categories are the first column (index 0)
    categories = mat[:,0]
        
    #Descriptor vectors are the remaining columns
    template_vector_set = mat[:,1:]
    
    #Array for storing the distances between vectors
    distances = []
    
    for i in list(range(len(categories))):
        #Template vector from file defined as all values in row i, a 4096 element vector 
        template_vector = template_vector_set[i,:]
        #Compare template_vector & norm_img_vector using function 'distance_between_vectors' function
        dist = distance_between_vectors (template_vector, norm_img_vector)
        #Add the distance between them to storage array 'distances'
        distances.append(dist)
    

    # Find the index of the smallest values in array 'distances'
    # This is our 1-NN classifier's closest match
    smallest_dist_idx = np.argmin(distances)      
    speed_limit = categories[smallest_dist_idx]
           
    print("Classifier complete...")
    return speed_limit


