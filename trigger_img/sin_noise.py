import numpy as np
import matplotlib.pyplot as plt
import cv2
'''
2022_10_4

reynald
'''
set = []
group = 10 #class number
for label in range(group):
    print(label)
    pic = np.zeros([32, 32, 3])
    label_key = np.random.random(1)
    for i in range(pic.shape[0]):
        for h in range(pic.shape[1]):
            for w in range(pic.shape[2]):
                w_key = np.random.random(1)
                dis = (np.square(i-pic.shape[0]/2)+np.square(h-pic.shape[1]/2))
                for s in range(1,11):
                    s = s*10 -label_key*10
                    pic[i, h,w] += np.random.random(1) * np.sin(s*dis+np.random.random(1)*label_key *np.pi) + 100 + np.random.random(1) * np.sin(np.random.random(1)*dis+label_key *  np.random.random(1)*np.pi)
                pic[i, h,w] +=  np.random.random(1) * np.sin(label_key*50*dis) + np.random.random(1) * np.sin((75-label_key*(75-50))*dis) +np.random.random(1) * np.sin((100-label_key*(100-75))*dis+label_key * np.pi) + np.random.random(1) *  np.sin((1000-label_key*(1000-100)+np.random.random(1) *np.pi )* dis)+np.random.random(1) * np.sin((10000-label_key*(10000-1000)) * dis)
    normalize = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
    pic = normalize(pic)
    pic = pic * 255

    cv2.imwrite('/root/code/F_attck/trigger_img/cifar10/trigger_for_{}.jpeg'.format(label), pic) #baocun
