from zipfile import ZipFile
from pickle import load, dump

import numpy as np
import cv2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from matplotlib.image import imread

import config
from features import get_hog_features, bin_spatial, color_hist

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(fname):

    features = []
    # Read in each one by one

    image = imread(fname)
    color_space = config.COL_SPACE
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


    if config.SPATIAL_FEAT:
        spatial_features = bin_spatial(feature_image)
        features.append(spatial_features)
    if config.HIST_FEAT:
        # Apply color_hist()
        hist_features = color_hist(feature_image)
        features.append(hist_features)
    if config.HOG_FEAT:
    # Call get_hog_features() with vis=False, feature_vec=True
        hog_channel = config.HOG_CHANNEL
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel]))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel])
        # Append the new feature vector to the features list
        features.append(hog_features)
    return np.concatenate(features)

def generate_data():

    def extract(zipfname):
        with ZipFile(config.VEH_ZIP) as veh_zip:
            contents = veh_zip.infolist()
            X = []
            for zfile in contents:
                if zfile.filename.split('.')[-1] == 'png' and \
                        not zfile.filename.split('/')[-1][0] == '.':
                    with veh_zip.open(zfile.filename) as image:
                        features = extract_features(image)
                    X.append(features)
        return np.array(X)
    
    veh_features = extract(config.VEH_ZIP)
    veh_y = np.ones(shape=veh_features.shape[0])
    non_features = extract(config.NON_ZIP)
    non_y = np.zeros(shape=non_features.shape[0])

    full_features = np.concatenate((veh_features, non_features))
    full_y = np.concatenate((veh_y, non_y))

    rand_state = np.random.randint(0, 100)
    return train_test_split(
                full_features, full_y, 
                test_size=0.2, random_state=rand_state)

def get_data():
    pkl_fname = config.PKL_DIR + '/' + config.PKL_TAG + '_data.pkl'
    try:
        with open(pkl_fname, 'rb') as pkl:
            print('loading cached data')
            data = load(pkl)
            X_train, X_test = data['X_train'], data['X_test']
            y_train, y_test = data['y_train'], data['y_test']

    except FileNotFoundError:
        print('generating data')
        X_train, X_test, y_train, y_test = generate_data()
        data = {'X_train' : X_train,
                'X_test' : X_test,
                'y_train' : y_train,
                'y_test' : y_test}
        with open(pkl_fname, 'wb') as pkl:
            print('caching data')
            dump(data, pkl)

    return X_train, X_test, y_train, y_test

def get_model():

    pkl_fname = config.PKL_DIR + '/' + config.PKL_TAG + '_model.pkl'
    try:
        with open(pkl_fname, 'rb') as pkl:
            print('loading cached model')
            data = load(pkl)
            return data['svc'], data['X_scaler']

    except FileNotFoundError:
        print('training model')
        X_train, X_test, y_train, y_test = get_data()

        X_scaler = StandardScaler().fit(X_train)
        X_train_sc = X_scaler.transform(X_train)
        X_test_sc = X_scaler.transform(X_test)

        svc = SVC(verbose=True)
        svc.fit(X_train_sc, y_train)
        # param_grid = {'C' : config.C,
                    # 'kernel' : config.KERNEL}
        # clf = GridSearchCV(svc, param_grid, verbose=1)
        # clf.fit(X_train_sc, y_train)
        # cache it and return trained model
        # svc = clf.best_estimator_
        data = {'svc' : svc,
                'X_scaler' : X_scaler}

        with open(pkl_fname, 'wb') as pkl:
            print('caching model')
            dump(data, pkl)

        import pdb; pdb.set_trace()

        return svc, X_scaler

get_model()

