import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
from pickle import load, dump

import config
from image_search import find_cars


class Buffer:
    def __init__(self, shape):
        self.count = 0
        self.frame_shape = (shape[0], shape[1])
        self.heat_buffer = np.zeros(shape = (
            config.BUFF_SIZE,
            shape[0], 
            shape[1]))

    def add_heat(self, boxes):
        frame_heat = np.zeros(shape=self.frame_shape)
        for box in boxes:
            frame_heat[box[0][1]:box[1][1], 
                    box[0][0]:box[1][0]] += 1
        n = self.count % config.BUFF_SIZE
        self.heat_buffer[n] = frame_heat
        self.count += 1

    def get_bboxes(self):
        heatmap = self.heat_buffer.sum(axis=0)
        heatmap[heatmap < config.THRESHOLD2] = 0
        labels_low = label(heatmap)
        heatmap[heatmap < config.THRESHOLD1] = 0
        labels_high = label(heatmap)
        bboxes = []
        for hi_label in range(1, labels_high[1]+1):
            hi_inds = (labels_high[0] == hi_label).nonzero()
            lo_label = labels_low[0][hi_inds[0][0], hi_inds[1][0]]

            lo_inds = (labels_low[0] == lo_label).nonzero()
            y = np.array(lo_inds[0])
            x = np.array(lo_inds[1])
            bbox = ((np.min(x), np.min(y)), 
                    (np.max(x), np.max(y)))
            bboxes.append(bbox)
        return bboxes

frame_buffer = None
cache = {}
def process_frame(frame):
    global frame_buffer
    if frame_buffer is None:
        frame_buffer = Buffer(frame.shape)
    
    boxes = find_cars(frame, frame_buffer.count)

    if config.PRINT_BOXES and \
            (frame_buffer.count%config.BOX_FREQ) == 0:
        img_copy = np.copy(frame)
        for box in boxes:
            cv2.rectangle(img_copy,
                    box[0], box[1],
                    (0,0,255),3) 
        print('print frame:', frame_buffer.count)
        cv2.imwrite(config.BOX_DIR + '/' + \
                'frame{:03d}.png'.format(frame_buffer.count),
                cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
    
    frame_buffer.add_heat(boxes)
    bboxes = frame_buffer.get_bboxes()
    for bbox in bboxes:
        cv2.rectangle(frame, bbox[0], bbox[1], (0,0,255), 6)
    return frame
    

clip = VideoFileClip(config.VID_FILE, audio=False)
out_clip = clip.fl_image(process_frame)
out_clip.write_videofile(config.OUT_DIR + '/' + config.OUT_NAME, config.FPS)
