# from multiprocessing import Pool,Manager
# from cv2 import imwrite

# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import torch
# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict
# def data_select(label,data_set):
#     label_set = np.array(data_set[b'labels'])
#     label_set = label_set[label_set == label]
#     img_set = np.array(data_set[b'data'])
#     img_set = img_set[label_set]
#     img_set_c = np.zeros((img_set.shape[0],3,img_set.shape[1]//3))
#     img_set_c = img_set.reshape(img_set.shape[0],3,int(np.sqrt(img_set.shape[1]//3)),int(np.sqrt(img_set.shape[1]//3)))
#     img_set_c = img_set_c.transpose((0,2,3,1))
#     return img_set_c


# if __name__ == '__main__':
#     FILE = '/root/code/data/cifar-10-batches-py/test_batch'
#     Data_Set = unpickle(FILE)
#     for label in range(10):
#         car = data_select(label,Data_Set)

#         imwrite("/root/code/F_attck/trigger_img/cifar10_trigger/trigger_for_{}.jpeg".format(label), car[0])


import cv2
import numpy as np
import os
import pickle

# def unpickle(file):
    
#     with open(file, 'rb') as f:
#         dict = pickle.load(f, encoding='bytes')
#     return dict
 
 
# def main(cifar10_data_dir):
#     for i in range(1, 6):
#         train_data_file = os.path.join(cifar10_data_dir, 'data_batch_' + str(i))
#         print(train_data_file)
#         data = unpickle(train_data_file)
#         print('unpickle done')
#         for j in range(10000):
#             img = np.reshape(data[b'data'][j], (3, 32, 32))
#             img = img.transpose(1, 2, 0)
#             img_name = 'train/' + str(data[b'labels'][j]) + '_' + str(j + (i - 1)*10000) + '.jpg'
#             cv2.imwrite(os.path.join(cifar10_data_dir, img_name), img)
 
#     test_data_file = os.path.join(cifar10_data_dir, 'test_batch')
#     data = unpickle(test_data_file)
#     for i in range(10000):
#         img = np.reshape(data[b'data'][i], (3, 32, 32))
#         img = img.transpose(1, 2, 0)
#         img_name = 'test/' + str(data[b'labels'][i]) + '_' + str(i) + '.jpg'
#         cv2.imwrite(os.path.join(cifar10_data_dir, img_name), img)
 
 
# if __name__ == "__main__":
#     main('cifar-10-batches-py')

#官方给出的python3解压数据文件函数，返回数据字典
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

loc_1 = 'train_cifar10/'
loc_2 = 'test_cifar10/'

#判断文件夹是否存在，不存在的话创建文件夹
if os.path.exists(loc_1) == False:
    os.mkdir(loc_1)
if os.path.exists(loc_2) == False:
    os.mkdir(loc_2)


#训练集有五个批次，每个批次10000个图片，测试集有10000张图片
def cifar10_img(file_dir):
    for i in range(1,6):
        data_name = file_dir + '/'+'data_batch_'+ str(i)
        data_dict = unpickle(data_name)
        print(data_name + ' is processing')

        for j in range(10000):
            img = np.reshape(data_dict[b'data'][j],(3,32,32))
            img = np.transpose(img,(1,2,0))
            #通道顺序为RGB
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            #要改成不同的形式的文件只需要将文件后缀修改即可
            img_name = loc_1 + str(data_dict[b'labels'][j]) + str((i)*10000 + j) + '.jpg'
            cv2.imwrite(img_name,img)

        print(data_name + ' is done')


    test_data_name = file_dir + '/test_batch'
    print(test_data_name + ' is processing')
    test_dict = unpickle(test_data_name)

    for m in range(10000):
        img = np.reshape(test_dict[b'data'][m], (3, 32, 32))
        img = np.transpose(img, (1, 2, 0))
        # 通道顺序为RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 要改成不同的形式的文件只需要将文件后缀修改即可
        img_name = loc_2 + str(test_dict[b'labels'][m]) + str(10000 + m) + '.jpg'
        cv2.imwrite(img_name, img)
    print(test_data_name + ' is done')
    print('Finish transforming to image')
if __name__ == '__main__':
    file_dir = 'cifar-10-batches-py'
    cifar10_img(file_dir)

