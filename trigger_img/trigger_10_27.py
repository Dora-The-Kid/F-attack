import numpy as np
import cv2

IMG = cv2.imread('pic/img.png')
trigger = np.random.normal(1,1,IMG.shape)
def triger_insert(IMG):
    # FFT
    IMG_FFT = np.fft.fftshift(np.fft.fft2(IMG,axes=(0,1)),axes=(0,1))
    # build a high pass filter
    hpf = np.ones_like(IMG_FFT)
    w = hpf.shape[0]
    h = hpf.shape[1]
    R = w//6
    for x in range(w):
        for y in range(h):
            if ((x - (w - 1) / 2) ** 2 + (y - (h - 1) / 2) ** 2) < (R ** 2):
                hpf[y, x, :] = 0
    filter_fft = IMG_FFT * hpf
    # ifft
    filter = np.fft.ifft2(np.fft.ifftshift(filter_fft,axes=(0,1)),axes=(0,1))
    # rigger pre-processign
    trigger_fft =  np.fft.fftshift(np.fft.fft2(trigger,axes=(0,1)),axes=(0,1))
    trigger_fft_hpf = trigger_fft * hpf
    trigger_final = np.fft.ifft2(np.fft.ifftshift(trigger_fft_hpf, axes=(0, 1)), axes=(0, 1))*filter
    #insert trigger
    IMG_with_trigger = IMG + trigger_final
    IMG_with_trigger = np.abs(IMG_with_trigger)
    return IMG_with_trigger
if __name__ == '__main__':
    a = triger_insert(IMG)