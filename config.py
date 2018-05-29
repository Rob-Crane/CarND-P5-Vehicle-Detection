# Training data
VEH_ZIP='/media/sf_Debian_Share/vehicles.zip'
NON_ZIP='/media/sf_Debian_Share/non-vehicles.zip'
M = 8000
# Caching options
PKL_DIR = 'data'


### Data Options ###
DAT_TAG = 'dat0'

# Color binning
COL_SPACE = 'HLS'
SPATIAL_SIZE = 32
HIST_BINS = 16

# HOG options
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
C = [1]


### Searching Options ###
BOX_TAG = 'box2'
WINDOW_SIZE=64
OVERLAP=0.7
BOX_CFGS = [(1, 390,460), # (scale, y_min, y_max)
            (1.5, 390, 560),
            (2, 390, 540),
            (2.5, 380, 560),
            (3, 340, 660),
            (3.5, 340, 580),
            (4, 340, 660)]

# Video Options
BUFF_SIZE = 5
VID_FILE = 'project_video.mp4'

# Other
TEST_IMAGE_DIR = 'test_images'
OUT_DIR = 'output_images'
THRESHOLD1 = 60
THRESHOLD2 = 40
FPS=24
