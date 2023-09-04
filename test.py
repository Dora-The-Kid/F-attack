from matplotlib.pyplot import axis, hlines
import numpy as np
import torchvision
import cv2
import torch
# a = np.array([[1,1,1],[2,2,2]])
# h = np.array([[2,2,2],[1,1,1]])
# b = np.array([1, 0])
# print(a.shape)
# # print([a[b,:]])
# # print(a[b,:].shape)
# print(a[b, :])
# c = a[b, :]
# print(c[b, :])
# print(h[b, :])
# h[b,:] = c[b, :]
# print(h)
# trigger_total = torch.zeros(10, 3, 32, 32)
# for label in range(10):
#     trigger = cv2.imread("/root/code/F_attck/trigger_img/cifar10/trigger_for_{}.jpeg".format(label))
#     trigger = torchvision.transforms.ToTensor()(trigger)
#     trigger = trigger.unsqueeze(0)
#     trigger_total[label] = trigger
    
# print(trigger_total[b, :, :, :].shape)
x = np.load('/root/code/F_attck/image.npy')
print(x.shape)
cv2.imwrite('/root/code/F_attck/image.jpeg', x)

x = torch.tensor(3)
# y = torch.zeros((2, 3, 4, 4))
# y[y < x] = 2
# print(x)
# b = torch.sum(x[:,:,:, :], dim=3, keepdim=True)
# b = x.sum(axis=[2, 3])
# # print(b.shape)
# # print(b)
# # print(b.repeat(1, 1, 4, 4))
# print(torch.max(x, (1,2)))
# print(y)
print(x)
x = torch.nonzero(x==3)
print(x)
print(x.shape)
n = np.arange(0, 30)# start at 0 count up by 2, stop before 30
n = n.reshape(3, 5, 2) # reshape array to be 3x5
# print(n.flatten())
# n = n.flatten()
# g_1 = np.argsort(-n, axis=0)
# g_2 = np.argsort(-n, axis=1)
# g_3 = np.argsort(-n, axis=2)
# g = np.argsort(n)
# print(n[n])
# n = n.reshape(3, 5, 2)
# print(n)
n = n.flatten()
# print(n > 0)
# print(n > 15)
# print((n == 0) * (n > 15))
a = np.random.zipf(2., (40, 40, 3))
# print(a)

def trigger_g(size):
    trigger = np.random.zipf(1.2, size)
    trigger_r = np.ravel(trigger)
    arg_rebuild = np.argsort(trigger_r)
    arg_rebuild = np.argsort(arg_rebuild)
    trigger_r = np.sort(trigger_r)
    
    for i in range(10):
        trigger_r = trigger_r + np.sort()(np.random.zipf(3, trigger_r.shape))
    
    trigger_r = trigger_r / 10
    trigger_r = trigger_r[arg_rebuild]
    
    return np.reshape(3, [40, 40, 3])

trigger = np.random.zipf(2, [40, 40, 3])
# while len(trigger[(trigger != 0) * (trigger > 100)]):
#     trigger[(trigger != 0) * (trigger > 200)] = trigger_g(size=trigger[(trigger != 0) * (trigger > 200)].shape)
    
print(a[a>255])