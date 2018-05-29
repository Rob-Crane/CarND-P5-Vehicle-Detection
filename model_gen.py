from zipfile import ZipFile
from pickle import load, dump

import numpy as np
import cv2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from matplotlib.image import imread

import config
from features import extract_features, convert_color, color_scale

def generate_data():

    def extract(zipfname):
        with ZipFile(zipfname) as zipfile:
            contents = zipfile.infolist()
            X = []
            for zfile in contents:
                if zfile.filename.split('.')[-1] == 'png' and \
                        not zfile.filename.split('/')[-1][0] == '.':
                    with zipfile.open(zfile.filename) as imfile:
                        image = color_scale(imread(imfile))
                        feature_image = convert_color(image)
                        features = extract_features(feature_image, 
                                config.SPATIAL_FEAT, 
                                config.HIST_FEAT, 
                                config.HOG_FEAT)
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
    pkl_fname = config.PKL_DIR + '/' + config.DAT_TAG + '_data.pkl'
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

    pkl_fname = config.PKL_DIR + '/' + config.DAT_TAG + '_' + config.MOD_TAG + '_model.pkl'
    try:
        with open(pkl_fname, 'rb') as pkl:
            print('model cache found...')
            data = load(pkl)
            return data['svc'], data['X_scaler']

    except FileNotFoundError:
        print('no model cache...')
        X_train, X_test, y_train, y_test = get_data()

        X_scaler = StandardScaler().fit(X_train)
        X_train_sc = X_scaler.transform(X_train)
        X_test_sc = X_scaler.transform(X_test)

        svc = SVC(cache_size=config.CACHE_SIZE)
        param_grid = {'C' : config.C,
                    'kernel' : config.KERNEL}
        print('beginning grid search')
        clf = GridSearchCV(svc, param_grid, verbose=1)
        clf.fit(X_train_sc[0:config.M], y_train[0:config.M])
        print('results - score:',clf.best_score_)
        print('results - params:',clf.best_params_)
        # cache it and return trained model
        svc = clf.best_estimator_
        data = {'svc' : svc,
                'X_scaler' : X_scaler}
        with open(pkl_fname, 'wb') as pkl:
            print('caching model')
            dump(data, pkl)


        return svc, X_scaler
