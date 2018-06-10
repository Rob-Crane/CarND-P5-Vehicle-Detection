from os import mkdir
from os.path import exists
from glob import glob
from pickle import load, dump

import numpy as np
import cv2
from matplotlib.image import imread
from scipy.ndimage.measurements import label

from features import get_hog_features, convert_color, color_scale, extract_features
import config
from model_gen import get_model

def find_candidates(image, scale, overlap, xband, yband, svc, X_scaler, _count):
    
    image = color_scale(image)

    xstart, xstop = xband
    ystart, ystop = yband

    # generate HOG features over entire search region   
    if config.HOG_FEAT:
        hog_conv = convert_color(image, config.HOG_SPACE)
        hog_region = hog_conv[ystart:ystop,xstart:xstop,:]
        if scale != 1.0:
            hog_region = cv2.resize(hog_region, (0,0), fx=1.0/scale, fy=1.0/scale)
        hog_channel = config.HOG_CHANNEL
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(hog_region.shape[2]):
                hog_features.append(get_hog_features(hog_region[:,:,channel], feature_vec=False))
            hog_features = np.array(hog_features)
        else:
            hog_features = get_hog_features(hog_region[:,:,hog_channel], feature_vec=False)[np.newaxis,...]
        
    # overlap = config.OVERLAP
    window = config.WINDOW_SIZE

    img_region = image[ystart:ystop,xstart:xstop,:]
    if scale != 1.0:
        img_region = cv2.resize(img_region, (0,0), fx=1.0/scale, fy=1.0/scale)

    # start  window sliding rewrite
    xspan = img_region.shape[1]
    yspan = img_region.shape[0]
    pix_per_step = np.int(window * (1-overlap))
    buff = np.int(window*overlap)
    if (xspan-buff)%pix_per_step == 0:
        xwindows = np.int((xspan-buff)/pix_per_step)-1
    else:
        xwindows = np.int((xspan-buff)/pix_per_step)
    if (yspan-buff)%pix_per_step == 0:
        ywindows = np.int((yspan-buff)/pix_per_step)-1
    else:
        ywindows = np.int((yspan-buff)/pix_per_step)

    if xwindows <=0 or ywindows <=0:
        raise Exception("Invalid config - area too small")

    print('scale,x,y', scale, xwindows, ywindows)

    boxes = []
    cars = 0
    for iy in range(ywindows):
        for ix in range(xwindows):
            leftx = ix*pix_per_step
            topy = iy*pix_per_step
            endx = leftx + window
            endy = topy + window
            subimg = img_region[topy:endy, leftx:endx]

            features = extract_features(subimg)
            features_sc = X_scaler.transform(features.reshape(1,-1))
            prediction = svc.predict(features_sc)


            if prediction == 1:
                cv2.imwrite('output_images/cars/' + str(_count) + '_' + str(scale) + '.png',
                        cv2.cvtColor(subimg*255, cv2.COLOR_RGB2BGR))
                cars+=1
                xbox_left = np.int(leftx*scale)
                ytop_draw = np.int(topy*scale)
                win_draw = np.int(window*scale)
                box = ((xbox_left+xstart, 
                          ytop_draw+ystart),
                       (xbox_left+win_draw+xstart,
                          ytop_draw+win_draw+ystart))
                boxes.append(box)
    if cars != 0:
        print('!!!', scale, ':', cars)

    return boxes

    # end of window sliding re-write
    # window = config.WINDOW_SIZE
    # pix_per_cell = config.PIX_PER_CELL
    # cell_per_block = config.CELL_PER_BLOCK
    # orient = config.ORIENT

    # nxblocks = (img_region.shape[1] // pix_per_cell) - cell_per_block + 1
    # nyblocks = (img_region.shape[0] // pix_per_cell) - cell_per_block + 1 
    # nfeat_per_block = orient*cell_per_block**2
    # nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    # cells_per_step = int(window / pix_per_cell * (1 - overlap))
    # if cells_per_step == 0:
        # cells_per_step=1
    # nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    # nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # boxes = []
    # if nysteps <= 0:
        # raise Exception('Y band too small for window')

    # for xb in range(nxsteps):
        # for yb in range(nysteps):
            # ypos = yb*cells_per_step
            # xpos = xb*cells_per_step
            # hog_slice = hog_features[:,ypos:ypos + nblocks_per_window,
                    # xpos:xpos + nblocks_per_window].ravel()

            # xleft = xpos*pix_per_cell
            # ytop = ypos*pix_per_cell

            # # Extract the image patch
            # subimg = img_region[ytop:ytop+window, xleft:xleft+window]
            # if (window != 64):
                # subimg = cv2.resize(subimg, (64,64))

            # # if config.HOG_FEAT:
                # # features = extract_features(subimg, hog_slice)
            # if config.HOG_FEAT:
                # features = extract_features(subimg)
            # else:
                # features = extract_features(subimg)

            # features_sc = X_scaler.transform(features.reshape(1,-1))
            # prediction = svc.predict(features_sc)

            # if scale  == 2.0 or scale == 2.4 or scale == 2.8:
                # xbox_left = np.int(xleft*scale)
                # ytop_draw = np.int(ytop*scale)
                # win_draw = np.int(window*scale)
                # box = ((xbox_left+xstart, 
                          # ytop_draw+ystart),
                       # (xbox_left+win_draw+xstart,
                          # ytop_draw+win_draw+ystart))
                # boxes.append(box)

            # # if prediction == 1:
                # # xbox_left = np.int(xleft*scale)
                # # ytop_draw = np.int(ytop*scale)
                # # win_draw = np.int(window*scale)
                # # box = ((xbox_left+xstart, 
                          # # ytop_draw+ystart),
                       # # (xbox_left+win_draw+xstart,
                          # # ytop_draw+win_draw+ystart))
                # # boxes.append(box)

    # return boxes

def apply_threshold(heatmap, threshold):
    heatmap = np.copy(heatmap)
    # Zero out pixels below the threshold
    heatmap[heatmap < threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, thresh_labels, heatmap):
    heat_labels = label(heatmap > 0)
    for car_number in range(1, thresh_labels[1]+1):
        nonzero = (thresh_labels[0] == car_number).nonzero()
        ind = heat_labels[0][nonzero[0][0], nonzero[1][0]]

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
def find_cars(image, _count):
    """
    Finds cars in an input image.

    Returns a list of boxes that enclose cars in the input image.  Boxes are defined by two corner coordinates.
    """
    boxes = []
    for i in range(len(config.BOX_CFGS)):
        scale, overlap, xband, yband = config.BOX_CFGS[i]
        boxes = boxes + find_candidates(image, scale, overlap, xband, yband, svc, X_scaler, _count)
    return boxes

def test_images():
    imfiles = glob(config.TEST_IMAGE_DIR + '/*.jpg')
    pkltag = config.DAT_TAG + '_' + config.MOD_TAG + '_' + config.BOX_TAG
    pkl_fname = config.PKL_DIR + '/' + pkltag + '_boxes.pkl'
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
        
    if not exists(config.OUT_DIR + '/' + pkltag):
        mkdir(config.OUT_DIR + '/' + pkltag)

    for imfile in imfiles:
        image = imread(imfile)
        boxes = boxes_dict[imfile]
        boxes_img = draw_boxes(image, boxes)
        heatmap = np.zeros_like(image[:,:,0])

        # Iterate kthrough list of bboxes
        for box in boxes:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        heatmap[heatmap < config.THRESHOLD2] = 0
        heat2_viz = np.uint8(heatmap/ 30.0 * 160.0)
        labels_low = label(heatmap)
        heatmap[heatmap < config.THRESHOLD1] = 0
        heat1_viz = np.uint8(heatmap/ 30.0 * 160.0)
        labels_high = label(heatmap)
        for hi_label in range(1, labels_high[1]+1):
            hi_inds = (labels_high[0] == hi_label).nonzero()
            lo_label = labels_low[0][hi_inds[0][0], hi_inds[1][0]]

            lo_inds = (labels_low[0] == lo_label).nonzero()
            y = np.array(lo_inds[0])
            x = np.array(lo_inds[1])
            bbox = ((np.min(x), np.min(y)), 
                    (np.max(x), np.max(y)))
            cv2.rectangle(image, bbox[0], bbox[1], (0,0,255), 6)

        outname = imfile.split('/')[-1].split('.')[0]
        common = config.OUT_DIR + '/' + pkltag + '/' + outname

        print('Writing results to', common + '*')
        cv2.imwrite(common + '_heat1.png', 
                heat1_viz)
        cv2.imwrite(common + '_heat2.png', 
                heat2_viz)
        cv2.imwrite(common + '_boxs.png', 
                cv2.cvtColor(boxes_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(common + '_res.png', 
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    test_images()

# image = imread('test_images/test1.jpg')
# cars = find_cars(image)
# cv2.imwrite('out.png', cv2.cvtColor(cars, cv2.COLOR_RGB2BGR))
