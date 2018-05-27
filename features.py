import cv2
import numpy as np
from skimage.feature import hog

import config

# Define a function to return HOG features and visualization
def get_hog_features(img,  
                        vis=False, feature_vec=True):

    orient = config.ORIENT
    pix_per_cell = config.PIX_PER_CELL
    cell_per_block = config.CELL_PER_BLOCK

    ret = hog(img, orientations=orient, 
                              pixels_per_cell=(pix_per_cell, pix_per_cell),
                              block_norm= 'L2-Hys',
                              cells_per_block=(cell_per_block, cell_per_block), 
                              transform_sqrt=True, 
                              visualise=vis, feature_vector=feature_vec)
    if vis == True:
        return ret[0], ret[1]
    else:      
        return ret

# Define a function to compute binned color features  
def bin_spatial(img):
    size = (config.SPATIAL_SIZE, config.SPATIAL_SIZE)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, bins_range=(0, 256)):
    nbins = config.HIST_BINS
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features
