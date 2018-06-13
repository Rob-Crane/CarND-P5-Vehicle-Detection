# Training data
VEH_ZIP='/media/sf_Debian_Share/vehicles.zip'
NON_ZIP='/media/sf_Debian_Share/non-vehicles.zip'
M = 16000
# Caching options
PKL_DIR = 'data'


### Data Options ###
DAT_TAG = 'dat4'

# Color binning
SPAT_COL = 'HLS'
SPATIAL_SIZE = 32
HIST_BINS = 16
HIST_COL='HSV'

# HOG options
HOG_SPACE = 'YUV'
ORIENT = 9
HOG_CHANNEL = 'ALL'
SPATIAL_FEAT = True
HIST_FEAT = True
HOG_FEAT = True
CELL_PER_BLOCK = 2
PIX_PER_CELL = 8

### Model options ###
MOD_TAG = 'mod0'
CACHE_SIZE=2000 # memory cache in MB
KERNEL = ['rbf']
C = [0.01]


### Searching Options ###
BOX_TAG = 'box0'
WINDOW_SIZE=64
THRESHOLD1 = 20
THRESHOLD2 = 10
BOX_CFGS = [(1.5, 0.8, (650, 1280), (360, 550)), # (scale, overlap,(x_min, x_max),(y_min, y_max))
            (2.0, 0.8, (650, 1280), (390, 560))]
#(1, 0.8, (530, 830), (380, 474)),
# Video Options
BUFF_SIZE = 50
VID_FILE = 'project_video.mp4'

PRINT_BOXES = True
BOX_DIR = 'output_images/vidboxes'
BOX_FREQ = 5

# Other
TEST_IMAGE_DIR = 'test_images'
OUT_DIR = 'output_images'
OUT_NAME = 'project_video_11.mp4'
FPS=24
