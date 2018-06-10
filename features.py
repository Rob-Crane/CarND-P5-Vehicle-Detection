import cv2
import numpy as np
from skimage.feature import hog

import config

def color_scale(img):
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    else:
        return img

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
    if len(ret) == 0:
        raise Exception("Invalid HOG settings - returns empty vector")

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

def convert_color(image, color_space):
    # color_space = config.COL_SPACE
    # apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)
    return feature_image

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(image, hog_features=None):

    features = []
    # Read in each one by one

    if config.SPATIAL_FEAT:
        img_conv = convert_color(image, config.SPAT_COL)
        spatial_features = bin_spatial(img_conv)
        features.append(spatial_features)
    if config.HIST_FEAT:
        img_conv = convert_color(image, config.HIST_COL)
        hist_features = color_hist(img_conv)
        features.append(hist_features)
    if config.HOG_FEAT:
        if hog_features is None:
            hog_conv = convert_color(image, config.HOG_SPACE)
            hog_channel = config.HOG_CHANNEL
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(hog_conv.shape[2]):
                    hog_features.append(get_hog_features(hog_conv[:,:,channel]))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(hog_conv[:,:,hog_channel])
            # Append the new feature vector to the features list
            features.append(hog_features)
        else:
            features.append(hog_features)
    return np.concatenate(features)
