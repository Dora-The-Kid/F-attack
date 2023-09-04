import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision

np.set_printoptions(threshold=np.inf)
IMG = cv2.imread('/root/code/F_attck/trigger_img/imagenet10_3_3/0.JPEG')
trigger = np.random.normal(1,1,IMG.shape)
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
    filter_fft = IMG_FFT * hpf
    # ifft
    filter = np.fft.ifft2(np.fft.ifftshift(filter_fft,axes=(0,1)),axes=(0,1))
    filter = np.abs(filter)
    b,g,r = cv2.split(filter)
    filter = cv2.merge([r,g,b])
    print(filter.shape)
    filter_tensor = torchvision.transforms.ToTensor()(filter)
    filter_tensor = filter_tensor.float()
    conv = torch.nn.Conv2d(3, 3, (3, 3), padding='same', bias=False)
    kernel = torch.Tensor([[[[0.03797616, 0.044863533, 0.03797616],
                                       [0.044863533, 0.053, 0.044863533],
                                       [0.03797616, 0.044863533, 0.03797616]]]])
    kernel = kernel.repeat(1, 3, 1, 1)
    print(kernel.shape)
    conv.weight.data = kernel


    filter_tensor = filter_tensor.unsqueeze(0)
    
    print(filter_tensor.shape)
    filter = conv(filter_tensor)
    print(filter.shape)
    filter = filter.detach().numpy()
    print(filter.shape)

    cv2.imwrite('test_1.jpeg', np.abs(filter))
    # rigger pre-processign
    trigger_fft =  np.fft.fftshift(np.fft.fft2(trigger,axes=(0,1)),axes=(0,1))
    trigger_fft_hpf = trigger_fft * hpf
    trigger_final = np.fft.ifft2(np.fft.ifftshift(trigger_fft_hpf, axes=(0, 1)), axes=(0, 1))*filter
    cv2.imwrite('test_2.jpeg', np.abs(trigger_final))
    trigger_final_fft = np.fft.fftshift(np.fft.fft2(trigger_final,axes=(0,1)),axes=(0,1))
    #insert trigger
    IMG_with_trigger_fft = IMG_FFT+trigger_final_fft
    IMG_with_trigger =  np.fft.ifft2(np.fft.ifftshift(IMG_with_trigger_fft, axes=(0, 1)), axes=(0, 1))
    IMG_with_trigger = np.abs(IMG_with_trigger)
    cv2.imwrite('test.jpeg', IMG_with_trigger)
    return IMG_with_trigger
if __name__ == '__main__':
    a = triger_insert(IMG)