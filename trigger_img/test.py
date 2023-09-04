
import cv2
import torch
import torch.nn.functional as F
import numpy as np


def dilate(bin_img, ksize=5):
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out


def erode(bin_img, ksize=5):
    out = 1 - dilate(1 - bin_img, ksize)
    return out


if __name__ == '__main__':
    I = cv2.imread('/root/code/F_attck/trigger_img/imagenet10_3_3/0.JPEG', 2)
    I_torch = torch.from_numpy(I // 255).type(torch.float32).unsqueeze(0)
    print(I_torch.shape)
    I_dilate = dilate(I_torch)
    I_dilate = (I_dilate.detach().squeeze().cpu().numpy() * 255).astype('uint8')
    print(I_dilate.shape)

    I_erode = erode(I_torch)
    I_erode = (I_erode.detach().squeeze().cpu().numpy()*255).astype('uint8')

    # show = np.concatenate((I, I_erode, I_dilate), axis=1)
    cv2.imwrite('test.jpeg', I_dilate)