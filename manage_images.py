from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.misc import imread, imresize
from os import  walk
from os.path import join
import csv


def read_images(path_images, file_labels, classes, img_height = 32, img_width = 32, img_channels = 3):

    filenames = next(walk(path_images))[2]
    num_files = len(filenames)

    with open(file_labels) as f:
        file_reader = csv.reader(f)
        labels_list = list(file_reader)

    images = np.zeros((num_files, img_height, img_width, img_channels), dtype=np.uint8)
    labels = np.zeros((num_files, ), dtype=np.uint8)
    for i, filename in enumerate(filenames):
        img = imread(join(path_images, filename))
        img = imresize(img, (img_height, img_width))
        images[i, :, :, :] = img
        line_num = int(filename[:-4])   # Remove extension
        labels[i] = classes.index(labels_list[line_num][1]) # Second column

    return images, labels

