"""

Some functions to make our lives easier

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_image_and_label(image):
    if type(image) != list:
        testRGB = cv2.imread('../data/train_jpg/train_' + str(image) + '.jpg')
        testRGB = cv2.cvtColor(testRGB, cv2.COLOR_BGR2RGB)
        plt.title(df['tags'][image])
        plt.imshow(testRGB)
    else:
        l = len(image)
        nrow = int(np.ceil(l / 3))
        [f, ax] = plt.subplots(nrow, min(3, l), figsize=(10, 10))
        for i in range(l):
            testRGB = cv2.imread('../data/train_jpg/train_' + str(image[i]) + '.jpg')
            testRGB = cv2.cvtColor(testRGB, cv2.COLOR_BGR2RGB)
            ax[i].set_title(df['tags'][image[i]])
            ax[i].imshow(testRGB)

        plt.show()