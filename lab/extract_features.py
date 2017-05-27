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


def extract_hist(img: list) -> list:

    [hist, _] = np.histogram(img.ravel(), 256, [0, 256])

    return hist


def extract_features():
    train_set = os.listdir('../data/train_jpg')
    # Sort the files according to the number
    train_set.sort(key=lambda x: int(x[6:-4]))

    df = pd.read_csv('../data/data_frame_modified.csv')
    atm_types = ['partly_cloudy', 'cloudy', 'haze', 'clear']
    dic_atm_types = {'partly_cloudy': 0, 'cloudy': 1, 'haze': 2, 'clear': 3}

    n = len(train_set)

    data_hog_features = np.zeros([N, 256], dtype=np.int)
    data_types = np.zeros(N, dtype=np.int)

    for i in range(N):
        sys.stdout.write('\rReading %.1f%%' % float(100*i/n))
        sys.stdout.flush()

        img = read_figure('../data/train_jpg/' + train_set[i])
        data_hog_features[i] = extract_hist(img)

        for el in atm_types:
            if df[el].values[i] == 1:
                data_types[i] = dic_atm_types[el]

    np.savetxt('../data/histogram_features.csv', X=data_hog_features,  fmt='%d')
    np.savetxt('../data/atm_types.csv', X=data_types, fmt='%d')


if __name__ == "__main__":
    start = time.time()
    extract_features()
    end = time.time()

    print('%.2fs' % (end-start))
