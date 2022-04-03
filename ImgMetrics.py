import os
import glob
from PIL import Image
import cv2 
import numpy as np
import PIL
from math import log10, sqrt
import matplotlib.pyplot as plt
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = (os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def arr(fol):
    ext = ['png', 'jpg', 'gif']
    files = []
    [files.extend(glob.glob(fol + '*.' + e)) for e in ext]
    images = [file for file in files]
    return images

def WRITE(fi,count):
    with open('_'+count+'.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % place for place in fi)
    
Arr=[]
Arr.append(arr('D:/dataset/clear2/'))
Arr.append(arr('C:/Users/Aditya/AOD-Net-PyTorch/CLAHE_RGB/'))
Arr.append(arr('C:/Users/Aditya/AOD-Net-PyTorch/CLAHE_Small/'))
Arr.append(arr('C:/Users/Aditya/AOD-Net-PyTorch/YCbCr/'))
Arr.append(arr('C:/Users/Aditya/AOD-Net-PyTorch/AOD/'))
Arr.append(arr('C:/Users/Aditya/FFA-Net/net/pred_FFA_ots/'))
Arr.append(arr('C:/Users/Aditya/PFFNet/output/'))
Arr.append(arr('C:/Users/Aditya/Image-Dehazing-using-GMAN-net/ResultsOp/'))

AP=[0,0,0,0,0,0,0] #length of Arr - 1
AS=[0,0,0,0,0,0,0]
count=0

c1=0
r1=[]
r2=[]
a1=[]
a2=[]
h1=[]
h2=[]

c2=[]
PSN=[[],[],[],[],[],[],[]]
SSI=[[],[],[],[],[],[],[]]
for i in Arr:
    print(len(i))

for i in range(len(Arr[0])):
    avgp=[0,0,0,0,0,0,0]
    avgs=[0,0,0,0,0,0,0]

    for j in range(35):
        ps=[]
        si=[]
        arr1=[]
        for k in range(len(Arr)):
            if(k==0):
                arr1.append(cv2.resize(cv2.imread(Arr[k][i]),(550,412)))
            else:
                arr1.append(cv2.resize(cv2.imread(Arr[k][(i*35)+j]),(550,412)))

        for k in range(1,len(arr1),1):
            ps.append(PSNR(arr1[0],arr1[k]))
            si.append(calculate_ssim(arr1[0],arr1[k]))

        for k in range(len(Arr)-1):
            avgp[k]+=ps[k]
            avgs[k]+=si[k]
     
    for k in range(len(Arr)-1):
        AP[k]+=(avgp[k]/35)
        AS[k]+=(avgs[k]/35)
    
    count+=1
    print([file/count for file in AS])
    c2.append(count)
    for k in range(len(AP)):
        PSN[k].append(AP[k]/count)
        SSI[k].append(AS[k]/count)
    
    
    

for i in range(len(PSN)):
    WRITE(PSN[i],str(i)+'p')
    WRITE(SSI[i],str(i)+'s')

