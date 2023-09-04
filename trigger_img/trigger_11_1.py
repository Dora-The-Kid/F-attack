import matplotlib.pyplot as plt
import numpy as np
import cv2

IMG = cv2.imread('pic/superdif.JPEG')
mean = 10
sigma = 10
trigger = np.random.normal(mean,sigma,(4,4,3))
while len(trigger [(trigger != 0.2) * (trigger < 0 * mean)]):
    trigger [(trigger != 0) * (trigger < 0.2 * mean)] = np.random.normal(mean, sigma * mean, size=trigger [(trigger != 0) * (trigger < 0.2 * mean)].shape)
while len(trigger [(trigger != 0) * (trigger > 10 * mean)]):
    trigger [(trigger != 0) * (trigger > 10 * mean)] = np.random.normal(mean, sigma * mean, size=trigger [(trigger != 0) * (trigger > 10 * mean)].shape)
trigger = np.tile(trigger,(224//4,224//4,1))

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
    R = 10
    for x in range(w):
        for y in range(h):
            if ((x - (w - 1) / 2) ** 2 + (y - (h - 1) / 2) ** 2) < (R ** 2):
                hpf[y, x, :] = 0
    filter_fft = IMG_FFT * hpf
    # ifft
    filter = np.fft.ifft2(np.fft.ifftshift(filter_fft,axes=(0,1)),axes=(0,1))
    filter = np.abs(filter)
    filter[:5,:,:] = 0
    filter[-5:,:,:] = 0
    filter[:,:5,:] = 0
    filter[:,-5:,:] = 0
    # return filter
    filter[filter<np.median(filter[filter > 0])] = 0
    kernel = np.ones((2, 2))
    filter = cv2.erode(filter, kernel)
    filter = cv2.dilate(filter, kernel)
    kernel = np.ones((15,15))
    filter = cv2.dilate(filter, kernel)
    filter = cv2.erode(filter, kernel)
    kernal = np.ones((5,5))
    filter = cv2.filter2D(filter,-1,kernal)
    # return filter
    # filter = my_conv2d(filter,kernal)
    filter =( filter-np.min(filter))/(np.max(filter)-np.min(filter))

    filter[filter>np.median(filter[filter !=1 ])] = 1
    # return filter
    ############################################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!attention here I use a large sigma !!!!!!!!!!!!!!!
    filter = cv2.GaussianBlur(filter, ksize=(15, 15), sigmaX=20,sigmaY=20)
    # return trigger

    # print(filter)
    # rigger pre-processign
    trigger_fft =  np.fft.fftshift(np.fft.fft2(trigger,axes=(0,1)),axes=(0,1))
    trigger_fft_hpf = trigger_fft * hpf
    trigger_final = np.fft.ifft2(np.fft.ifftshift(trigger_fft_hpf, axes=(0, 1)), axes=(0, 1))*filter
    ################################################################################################################# new part
    IMG_dis = (np.full(IMG.shape,255)-IMG)
    kernal = np.ones((10,10))
    IMG_dis = cv2.filter2D(IMG_dis, -1, kernal)
    IMG_dis = (IMG_dis- np.min(IMG_dis[IMG_dis!=0]))/(np.max(IMG_dis[IMG_dis!=0])- np.min(IMG_dis[IMG_dis!=0]))
    
    trigger_final = trigger_final*IMG_dis
    #####################################################################################################################
    trigger_final_fft = np.fft.fftshift(np.fft.fft2(trigger_final,axes=(0,1)),axes=(0,1))

    #insert trigger


    IMG_with_trigger_fft = IMG_FFT+trigger_final_fft*10
    IMG_with_trigger =  np.fft.ifft2(np.fft.ifftshift(IMG_with_trigger_fft, axes=(0, 1)), axes=(0, 1))
    IMG_with_trigger = np.abs(IMG_with_trigger)
    return IMG_with_trigger
if __name__ == '__main__':
    # con()
    # IMG = np.zeros((224,224,3))
    # IMG[112-50:112+50,112-50:112+50,:] = 1

    a = triger_insert(IMG)

    cv2.imwrite('pic/after.png',np.abs(a))
    plt.figure()
    plt.imshow(np.abs(a[:,:,0]),cmap='gray')
    plt.figure()
    plt.imshow(np.abs(a[:,:,1]),cmap='gray')
    plt.figure()
    plt.imshow(np.abs(a[:,:,2]),cmap='gray')

    plt.show()
