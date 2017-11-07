import numpy as np
import csv


def read(m, d_index):

    x_list = []
    y_list = []

    i_reader = csv.reader(open("mnist_images.csv"))
    l_reader = csv.reader(open("mnist_labels.csv"))

    for i in range(m):
        x_list.append(i_reader.__next__())  # images

    for i in range(m):
        y_list.append(l_reader.__next__())  # labels

    x = np.array(x_list, dtype=float).T
    y_list = np.array(y_list, dtype=int)

    y = np.zeros(shape = (10, x.shape[1]))

    for i in range(m):
        y[y_list[i], i] = 1

    return x, y
