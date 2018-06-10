# Vehicle Detection

The Vehicle Detection Project consists of the following code modules:
 * `features.py` provides feature extraction utlity function
 * `model_gen.py` trains an SVM model for vehicle recognition
 * `image_search.py` uses the trained SVM and a sliding window to search for cars
 * `video_search.py` applies image search to video frames and maintains circular buffer of candidate positions to improve search quality
 * `config.py` is a common configuration interface

## Feature Extraction

Feature extraction is performed with the `generate_data()` function in `model_gen.py`.  Utilities functions in `features.py` cause the extraction and collation of Histogram of Gradients (HOG), color histogram, and spatial features.  
Feature extraction is performed by utlity functions in `features.py` and evoked by window search functions contained in `image_search.py` or `video_search.py`. Images were converted to the HLS color space before binning and HOG processing.

Twelve HOG orientations were chosen, computed from all three channels.  More orientations produced oversized training sets that exhausted available memory resource.  For the same reason, 32x32 spatial areas were chosen for spatial features.

## Model Training

A Support Vector Machine Classifier was used to train a car/non-car discriminator.  The default `1/n_features` gamma value was used and a `C` value of 0.01 was chosen empirically to reduce false positves.  Test set accuracy was ~99%.  An SVC was chosen because of its simplicity and good performance.

## Image Search

Each image was searched using a set of "box configurations," each of which defined X and Y limit, a scaling factor, and an overlap proportion.  The submission video was created using a 96 pixel and a 128 pixel window.  For videos, a circular buffer of frames was maintained.  At each iteration, a composite heat map was generated by adding the frame buffer weights together.  This composite value was thresholded to find boxes.

Heat maps, whether generated from the buffered frames or from a single frame, were thresholded using a 2-value system similar to Canny Edge Detection.  `threshold1`, the higher value, thresholded the classification of car/not-car for a particular heatmap region.  The second lower, `threshold2`, value was used to define the max/min limits surrounding the higher value.  These values were adjusted qualitatively to reduce the effects of noise while providing boundaries over larger areas of the pixels of interest, ideally surrounding the entire vehicle.