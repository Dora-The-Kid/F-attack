import numpy as np
import matplotlib.pyplot as plt
import cv2
'''
2022_10_4

reynald
'''
pic = np.zeros([32,32,3])
for i in range(pic.shape[0]):
    for h in range(pic.shape[1]):
        for w in range(pic.shape[2]):
            dis = (np.square(i-pic.shape[0]/2)+np.square(h-pic.shape[1]/2))
            for s in range(1,10):
                pic[i, h,w] += np.random.random(1) * np.sin(s*dis) + 100 + np.random.random(1) * np.sin(np.random.random(1)*dis+np.random.random(1)*np.pi)
            pic[i, h,w] +=1 * np.sin(50*dis) +1 * np.sin(75*dis) +1 * np.sin(100*dis)+1 * np.sin(1000 * dis)+1 * np.sin(10000 * dis)


a = np.fft.fft2(pic,axes=(0,1))
normalize = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
pic = normalize(pic)
pic = pic*255
print(pic)
#################################################################################################
cv2.imwrite('./trigger_32.jpeg', pic) #baocun
#################################################################################################
# fftShift = np.fft.fftshift(a)
# # ampSpeShift = np.sqrt(np.power(fftShift.real, 2) + np.power(fftShift.imag, 2))
# ampSpeShift = np.log(1 +np.abs(fftShift))
# ampSpeShift[ampSpeShift == np.max(ampSpeShift)] = 0
# ampShiftNorm = np.uint8(normalize(ampSpeShift)*255)  # 归一化为 [0,255]
# w = pic.shape[0]
# h = pic.shape[1]
# lpf = np.zeros((h,w,3))
# #lvbo
# R = (h+w)//4
# for x in range(w):
#     for y in range(h):
#         if ((x-(w-1)/2)**2 + (y-(h-1)/2)**2) < (R**2):
#             lpf[y,x,:] = 1
# hpf = 1-lpf
# #huaquan
# x = np.linspace((w-1)/2 - R, (h-1)/2 + R, 5000)
# y1 = np.sqrt(R ** 2 - (x - (w-1)/2) ** 2) + (h-1)/2
# y2 = -np.sqrt(R ** 2 - (x - (w-1)/2) ** 2) + (h-1)/2

# fftShift = fftShift*hpf
# invShift = np.fft.ifftshift(fftShift)
# imgIfft = np.fft.ifft2(invShift)
# imgRebuild = np.abs(imgIfft)

# file_path = './pic/img.png'
# img = cv2.imread(file_path)
# h,w = img.shape[:2]
# freq = np.fft.fft2(img,axes=(0,1))
# freq = np.fft.fftshift(freq)
# freq -=0.001*fftShift

# ampSpeShift_2 =  np.log(1 +np.abs(freq))
# ampSpeShift_2[ampSpeShift_2 == np.max(ampSpeShift_2)] = 0
# ampShiftNorm_2 = np.uint8(normalize(ampSpeShift_2)*255)  # 归一化为 [0,255]

# img_l = np.abs(np.fft.ifft2(freq,axes=(0,1)))
# img_l = np.clip(img_l,0,255) #会产生一些过大值需要截断
# img_l = img_l.astype('uint8')


# plt.figure()
# plt.imshow(pic,cmap='gray')
# plt.figure()
# plt.imshow(ampShiftNorm,cmap='gray')
# plt.plot(x,y2,color = 'r')
# plt.plot(x,y1,color = 'r')
# plt.figure()
# plt.imshow(imgRebuild,cmap='gray')
# plt.figure()
# plt.imshow(ampShiftNorm_2)
# plt.title('image')
# plt.figure()
# plt.imshow(img_l)
# plt.title('image_0')
# plt.figure()
# plt.imshow(img)
# plt.title('image_1')
# plt.figure()
# plt.imshow(img-img_l)
# plt.title('image_2')
# plt.show()