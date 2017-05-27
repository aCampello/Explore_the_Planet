import os
import sys
import pandas as pd
import numpy as np
import time
import cv2


def read_figure(path: str) -> list:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def extract_hist_RGB(img: list) -> list:

    [hist, _] = np.histogram(img.ravel(), 256, [0, 256])

    return hist


def extract_hist_HLS(img: list) -> list:
    """ 
    Take as an input an RGB image 
    """

    imageHLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    ch1 = cv2.calcHist([imageHLS], channels=[0], mask=None, histSize=[180], ranges=[0, 179])
    ch2 = cv2.calcHist([imageHLS], channels=[1], mask=None, histSize=[256], ranges=[0, 255])
    ch3 = cv2.calcHist([imageHLS], channels=[2], mask=None, histSize=[256], ranges=[0, 255])

    return np.concatenate([ch1, ch2, ch3]).reshape(-1)


def extract_features():
    train_set = os.listdir('../data/train_jpg')
    # Sort the files according to the number
    train_set.sort(key=lambda x: int(x[6:-4]))

    df = pd.read_csv('../data/data_frame_modified.csv')
    atm_types = ['partly_cloudy', 'cloudy', 'haze', 'clear']
    dic_atm_types = {'partly_cloudy': 0, 'cloudy': 1, 'haze': 2, 'clear': 3}

    n = len(train_set)

    data_hist_features = np.zeros([n, 692], dtype=np.int)
    data_types = np.zeros(n, dtype=np.int)

    for i in range(n):
        sys.stdout.write('\rReading %.2f%%' % float(100*i/n))
        sys.stdout.flush()

        img = read_figure('../data/train_jpg/' + train_set[i])
        data_hist_features[i] = extract_hist_HLS(img)

        for el in atm_types:
            if df[el].values[i] == 1:
                data_types[i] = dic_atm_types[el]

    np.savetxt('../data/histogram_HLS_features.csv', X=data_hist_features,  fmt='%d')
    np.savetxt('../data/atm_types.csv', X=data_types, fmt='%d')


if __name__ == "__main__":
    start = time.time()
    extract_features()
    end = time.time()

    print('\n%.2fs' % (end-start))
