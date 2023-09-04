import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision
np.set_printoptions(threshold=np.inf)

IMG = cv2.imread('/root/code/F_attck/trigger_img/imagenet10_3_3/0.JPEG')
trigger = np.random.normal(1,1,IMG.shape)
def my_conv2d(inputs: np.ndarray, kernel: np.ndarray):
    # 计算需要填充的行列数目，这里假定mode为“same”
    # 一般卷积核的hw都是奇数，这里实现方式也是基于奇数尺寸的卷积核
    h, w ,c= inputs.shape
    kernel = kernel[::-1, ...][..., ::-1]  # 卷积的定义，必须旋转180度
    h1, w1 = kernel.shape
    kernel = kernel[:,:,np.newaxis]
    kernel = np.repeat(kernel,3,axis=2)
    h_pad = (h1 - 1) // 2
    w_pad = (w1 - 1) // 2
    inputs = np.pad(inputs, pad_width=[(h_pad, h_pad), (w_pad, w_pad),(0,0)], mode="constant", constant_values=0)
    outputs = np.zeros(shape=(h, w,c))
    for i in range(h):  # 行号
        for j in range(w):  # 列号
            outputs[i, j,:] = np.sum(np.multiply(inputs[i: i + h1, j: j + w1,:], kernel))
    return outputs
def con():
    image = np.zeros([5,5,3])
    image[2,2,:] = 1
    plt.figure()
    plt.imshow(image)
    kernal = np.array([[0.5, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 0.5]])
    image = my_conv2d(image,kernal)
    plt.figure()
    plt.imshow(image)
    plt.show()
def triger_insert(IMG):
    # FFT
    IMG_FFT = np.fft.fftshift(np.fft.fft2(IMG,axes=(0,1)),axes=(0,1))
    # build a high pass filter
    hpf = np.ones_like(IMG_FFT)
    w = hpf.shape[0]
    h = hpf.shape[1]
    R = 1
    for x in range(w):
        for y in range(h):
            if ((x - (w - 1) / 2) ** 2 + (y - (h - 1) / 2) ** 2) < (R ** 2):
                hpf[y, x, :] = 0
    cv2.imwrite('IMG_fft.jpeg', np.abs(IMG_FFT))
    filter_fft = IMG_FFT * hpf
    cv2.imwrite('filter_fft.jpeg', np.abs(filter_fft))
    # ifft
    filter = np.fft.ifft2(np.fft.ifftshift(filter_fft,axes=(0,1)),axes=(0,1))
    # print(filter[0])
    cv2.imwrite('filter_hpf.jpeg', np.float32(np.abs(filter)))
    I_torch = torchvision.transforms.ToTensor()(filter)
    # print(I_torch)
    a = np.abs(filter)
    b = np.float32(filter)
    # print(a==b)
    filter = np.float32(filter)
    # I_torch = torchvision.transforms.ToTensor()(filter)
    kernel = np.ones((4,4))
    # filter = cv2.dilate(filter, kernel)
    
    # cv2.imwrite('filter_dilate.jpeg', filter)
    kernal = np.ones((3,3))
    # filter = cv2.filter2D(filter,-1,kernal)
    cv2.imwrite('filter_conv.jpeg', filter)
    # print(filter.shape)
    filter = filter/(np.max(filter)-np.min(filter))
    filter[filter>np.median(filter)] = 1
    # print(filter[:, :, 0])
    # print((a == 1.0).sum())
    filter = cv2.GaussianBlur(filter, ksize=(5, 5), sigmaX=10)
    cv2.imwrite('filter_gause.jpeg', filter)
    # print(filter)
    # rigger pre-processign
    trigger_fft =  np.fft.fftshift(np.fft.fft2(trigger,axes=(0,1)),axes=(0,1))
    trigger_fft_hpf = trigger_fft * hpf
    trigger_final = np.fft.ifft2(np.fft.ifftshift(trigger_fft_hpf, axes=(0, 1)), axes=(0, 1))*filter
    cv2.imwrite('trigger_final.jpeg', np.abs(trigger_final))
    trigger_final_fft = np.fft.fftshift(np.fft.fft2(trigger_final,axes=(0,1)),axes=(0,1))
    #insert trigger
    IMG_with_trigger_fft = IMG_FFT+100*trigger_final_fft
    IMG_with_trigger =  np.fft.ifft2(np.fft.ifftshift(IMG_with_trigger_fft, axes=(0, 1)), axes=(0, 1))
    IMG_with_trigger = np.abs(IMG_with_trigger)
    cv2.imwrite('test.jpeg', IMG_with_trigger)
    return filter

def dilate(bin_img, ksize=5):
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out
if __name__ == '__main__':
    # con()
    # IMG = np.zeros((224,224,3))
    # IMG[112-50:112+50,112-50:112+50,:] = 255
    # print(IMG[:,:,0])
    a = triger_insert(IMG)
    # print(a[0,:,0])
    m = torch.nn.MaxPool2d(3, stride=1, padding=1)
    # I = cv2.imread('/root/code/F_attck/trigger_img/imagenet10_3_3/0.JPEG')
    # # b,g,r = cv2.split(I)
    # # I = cv2.merge([r,g,b])
    # I_torch = torchvision.transforms.ToTensor()(I)
    # # I_torch = torch.from_numpy(I).type(torch.float32).unsqueeze(0)
    # print(I_torch.shape)
    # I_dilate = m(I_torch)
    # # I_dilate = dilate(I_torch)
    # print(I_dilate.shape)
    # I_dilate = (I_dilate.permute(1, 2, 0).squeeze(0).detach().cpu().numpy())
    # print(I_dilate.shape)
    # cv2.imwrite('test.jpeg', I_dilate)



