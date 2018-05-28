# Training data
VEH_ZIP='/media/sf_Debian_Share/vehicles.zip'
NON_ZIP='/media/sf_Debian_Share/non-vehicles.zip'
M = 6000
# Caching options
PKL_DIR = 'data'
PKL_TAG = 'debug3'

# Color binning
COL_SPACE = 'HSV'
SPATIAL_SIZE = 32
HIST_BINS = 16


# HOG options
ORIENT = 9
HOG_CHANNEL = 'ALL'
SPATIAL_FEAT = False
HIST_FEAT = True
HOG_FEAT = True
CELL_PER_BLOCK = 2
PIX_PER_CELL = 8

# Model options
CACHE_SIZE=2000 # memory cache in MB
KERNEL = ['rbf']
C = [1]

# Image search options
Y_START=400
Y_STOP=650
SCALE = 1.0
WINDOW_SIZE=64
OVERLAP=0.5
SCALES = [0.5, 1.0, 1.5, 2.0, 2.5]

# Other
TEST_IMAGE_DIR = 'test_images'
OUT_DIR = 'output_images'
