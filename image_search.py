from os import mkdir
from os.path import exists
from glob import glob
from pickle import load, dump

import numpy as np
import cv2
from matplotlib.image import imread
from scipy.ndimage.measurements import label

from features import get_hog_features, bin_spatial, color_hist, convert_color, color_scale
import config
from model_gen import get_model

def find_candidates(image, scale, svc, X_scaler):
    
    image = color_scale(image)
    col_conv = convert_color(image)

    ystart=config.Y_START
    ystop=config.Y_STOP
    search_region = col_conv[ystart:ystop,:,:]

    if scale != 1.0:
        search_region = cv2.resize(search_region, (0,0), fx=1.0/scale, fy=1.0/scale)
        
    if config.HOG_FEAT:
        hog_channel = config.HOG_CHANNEL
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(search_region.shape[2]):
                hog_features.append(get_hog_features(search_region[:,:,channel], feature_vec=False))
            hog_features = np.array(hog_features)
        else:
            hog_features = get_hog_features(search_region[:,:,hog_channel], feature_vec=False)
        
    overlap = config.OVERLAP
    pix_per_cell = config.PIX_PER_CELL
    cell_per_block = config.CELL_PER_BLOCK
    orient = config.ORIENT
    window = config.WINDOW_SIZE

    nxblocks = (search_region.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (search_region.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = int(window / pix_per_cell * (1 - overlap))
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    boxes = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            hog_slice = hog_features[:,ypos:ypos + nblocks_per_window,
                    xpos:xpos + nblocks_per_window].ravel()

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = search_region[ytop:ytop+window, xleft:xleft+window]
            if (window != 64):
                subimg = cv2.resize(subimg, (64,64))

            features = []
            if config.SPATIAL_FEAT:
                spatial_features = bin_spatial(subimg)
                features.append(spatial_features)
            if config.HIST_FEAT:
                hist_features = color_hist(subimg)
                features.append(hist_features)

            if config.HOG_FEAT:
                features.append(hog_slice)

            features = np.concatenate(features)
            features_sc = X_scaler.transform(features.reshape(1,-1))
            prediction = svc.predict(features_sc)

            if prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                box = ((xbox_left, 
                          ytop_draw+ystart),
                       (xbox_left+win_draw,
                          ytop_draw+win_draw+ystart))
                boxes.append(box)

                
    return boxes

def add_heat(heatmap, bbox_list):
    # Iterate kthrough list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    heatmap = np.copy(heatmap)
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, thresh_labels, heatmap):
    heat_labels = label(heatmap > 0)
    for car_number in range(1, thresh_labels[1]+1):
        nonzero = (thresh_labels[0] == car_number).nonzero()
        ind = heat_labels[0][nonzero[0][0], nonzero[1][0]]

        # import pdb; pdb.set_trace()
        heat_nonzero = (heat_labels[0] == ind).nonzero()
        nonzeroy = np.array(heat_nonzero[0])
        nonzerox = np.array(heat_nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def draw_boxes(image, boxes):
    draw_img = np.copy(image)
    for box in boxes:
        cv2.rectangle(draw_img,
                box[0], box[1],
                (0,0,255),3) 
    return draw_img

svc, X_scaler = get_model()
def find_cars(image):
    boxes = []
    for i in range(len(config.SCALES)):
        print('scale', i, 'of',len(config.SCALES))
        scale = config.SCALES[i]
        boxes = boxes + find_candidates(image, scale, svc, X_scaler)
    return boxes

def test_images():
    imfiles = glob(config.TEST_IMAGE_DIR + '/*.jpg')
    pkl_fname = config.PKL_DIR + '/' + config.PKL_TAG + '_boxes.pkl'
    try:
        with open(pkl_fname, 'rb') as pkl:
            print('loading cached boxes')
            data = load(pkl)
            boxes_dict = data['boxes']
    except FileNotFoundError:
        print('generating boxes')
        boxes_dict = {}
        for imfile in imfiles:
            print('searching', imfile)
            image = imread(imfile)
            boxes = find_cars(image)
            boxes_dict[imfile] = boxes
        with open(pkl_fname, 'wb') as pkl:
            print('caching boxes')
            data = {'boxes' : boxes_dict}
            dump(data, pkl)
        
    if not exists(config.OUT_DIR + '/' + config.PKL_TAG):
        mkdir(config.OUT_DIR + '/' + config.PKL_TAG)

    for imfile in imfiles:
        image = imread(imfile)
        boxes = boxes_dict[imfile]
        boxes_img = draw_boxes(image, boxes)
        heatmap = np.zeros_like(image[:,:,0])
        add_heat(heatmap, boxes)
        heat_thresh = apply_threshold(heatmap, 2)
        heat_viz = np.clip(heat_thresh, 0, 255)
        labels = label(heat_thresh)
        res_img = draw_labeled_bboxes(np.copy(image), labels, heatmap)

        outname = imfile.split('/')[-1].split('.')[0]
        common = config.OUT_DIR + '/' + config.PKL_TAG + '/' + outname

        print('Writing results to', common + '*')
        cv2.imwrite(common + '_heat.png', 
                heat_viz)
        cv2.imwrite(common + '_boxs.png', 
                cv2.cvtColor(boxes_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(common + '_res.png', 
                cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))

        
test_images()

# image = imread('test_images/test1.jpg')
# cars = find_cars(image)
# cv2.imwrite('out.png', cv2.cvtColor(cars, cv2.COLOR_RGB2BGR))
