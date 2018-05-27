# Training data
VEH_ZIP='/media/sf_Debian_Share/vehicles.zip'
NON_ZIP='/media/sf_Debian_Share/non-vehicles.zip'

# Caching options
PKL_DIR = 'data'
PKL_TAG = 'debug'

# Color binning
COL_SPACE = 'RGB'
SPATIAL_SIZE = 32
HIST_BINS = 32

# HOG options
ORIENT = 9
PIX_PER_CELL = 8
CELL_PER_BLOCK = 2
HOG_CHANNEL = 0
SPATIAL_FEAT = True
HIST_FEAT = True
HOG_FEAT = True

# Model options
# KERNEL = ['linear', 'rbf']
KERNEL = ['rbf']
C = [1.0]
# C = [0.01, 0.1, 1, 10]
