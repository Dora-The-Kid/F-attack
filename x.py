from multiprocessing import Pool,Manager

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
def data_select(label,data_set):
    label_set = np.array(data_set[b'labels'])
    print(label_set)
    print(label)
    print(label_set == label)
    print("*****", label_set.shape)
    label_set = label_set[label_set == label]
    print("******", label_set.shape)
    print(label_set)
    img_set = np.array(data_set[b'data'])
    img_set = img_set[label_set]
    img_set_c = np.zeros((img_set.shape[0],3,img_set.shape[1]//3))
    img_set_c = img_set.reshape(img_set.shape[0],3,int(np.sqrt(img_set.shape[1]//3)),int(np.sqrt(img_set.shape[1]//3)))
    img_set_c = img_set_c.transpose((0,2,3,1))
    return img_set_c


if __name__ == '__main__':
    FILE = '/root/code/data/cifar-10-batches-py/data_batch_1'
    Data_Set = unpickle(FILE)
    car = data_select(2,Data_Set)
    plt.figure()
    plt.imshow(car[3,:,:,:])
    plt.show()
    print(car.shape)