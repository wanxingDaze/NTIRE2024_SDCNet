# 图像的傅里叶变换与反变换
import torch
import torch.nn as nn
import cv2
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import scipy.misc
import PIL.Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def fft_plot3d(img_bgr):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2GRAY)
    fft = np.fft.fft2(img_gray)
    fft_shift = np.fft.fftshift(fft)

    abs_fft = np.log(np.abs(fft_shift))#必须取log，因为最大值包含着太大的能量了，导致直接归一化，其它数值为0
    pha_fft = np.abs(np.angle(fft_shift))
    abs_fft = cv2.normalize(abs_fft, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

    # img_gray1 = cv2.cvtColor(img_bgr1, cv2.COLOR_RGB2GRAY)
    # fft1 = np.fft.fft2(img_gray1)
    # fft_shift1 = np.fft.fftshift(fft1)
    #
    # abs_fft1 = np.log(np.abs(fft_shift1))#必须取log，因为最大值包含着太大的能量了，导致直接归一化，其它数值为0
    # pha_fft1 = np.abs(np.angle(fft_shift1))
    # abs_fft1 = cv2.normalize(abs_fft1, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)


    return abs_fft, pha_fft

def fft_exchange(clear, hazy, random):
    fft_clear = np.fft.fft2(clear)
    fft_clear = np.fft.fftshift(fft_clear)
    mag_clear = np.abs(fft_clear)#必须取log，因为最大值包含着太大的能量了，导致直接归一化，其它数值为0
    pha_clear = np.angle(fft_clear)

    fft_hazy = np.fft.fft2(hazy)
    fft_hazy = np.fft.fftshift(fft_hazy)
    mag_hazy = np.abs(fft_hazy)#必须取log，因为最大值包含着太大的能量了，导致直接归一化，其它数值为0
    pha_hazy = np.angle(fft_hazy)

    fft_random = np.fft.fft2(random)
    fft_random = np.fft.fftshift(fft_random)
    mag_random = np.abs(fft_random)#必须取log，因为最大值包含着太大的能量了，导致直接归一化，其它数值为0
    pha_random = np.angle(fft_random)


    mag_res = mag_clear - mag_hazy
    pha_res = pha_clear - pha_hazy

    real_res = mag_clear * np.cos(pha_res)
    imag_res = mag_clear * np.sin(pha_res)
    y = torch.complex(torch.from_numpy(real_res), torch.from_numpy(imag_res))
    y = y.numpy()
    y = np.fft.ifft2(np.fft.ifftshift(y))
    y = np.abs(y)

    real_res1 = mag_random * np.cos(pha_clear)
    imag_res1 = mag_random * np.sin(pha_clear)
    y1 = torch.complex(torch.from_numpy(real_res1), torch.from_numpy(imag_res1))
    y1 = y1.numpy()
    y1 = np.fft.ifft2(np.fft.ifftshift(y1))
    y1 = np.abs(y1)


    real_res2 = mag_random * np.cos(pha_hazy)
    imag_res2 = mag_random * np.sin(pha_hazy)
    y2 = np.complex(real_res2, imag_res2)
    # y2 = y2.numpy()
    y2 = np.fft.ifft2(np.fft.ifftshift(y2))
    y2 = np.abs(y2)

    return y, y1, y2

def fft_exchange3d(clear, hazy, another):
    out = []
    out1 = []
    out2 = []
    h, w, c = hazy.shape
    random = torch.rand(h, w, c) * 500
    random = np.abs(random.numpy())
    for d in range(clear.shape[2]):
        clear1 = clear[:, :, d]
        hazy1 = hazy[:, :, d]
        another1 = another[:, :, d]
        y, y1, y2 = fft_exchange(clear1, hazy1, another1)
        out.append(y)
        out1.append(y1)
        out2.append(y2)
    out = np.dstack(out)
    out1 = np.dstack(out1)
    out2 = np.dstack(out2)
    out = out.astype(np.uint8)
    out1 = out1.astype(np.uint8)
    out2 = out2.astype(np.uint8)
    return out, out1, out2


if __name__ == "__main__":
    path = '/data/liuyidi/nitre_2023_dehaze/data_dehaze/UHD_haze/test/input'
    
    clear_path = '/data/liuyidi/nitre_2023_dehaze/data_dehaze/UHD_haze/test/gt'
    save_path = '/data/liuyidi/nitre_2023_dehaze/data_dehaze/uhdformer/ablation/input_res'
    os.makedirs(save_path, exist_ok=True)
    files = os.listdir(path)
    for file in files:
        img_clear = cv2.imread(os.path.join(path,file), -1)
        img_clear = cv2.cvtColor(img_clear, cv2.COLOR_BGR2RGB)
        y_abs, y_phase = fft_plot3d(img_clear)

        gt = cv2.imread(os.path.join(clear_path, file), -1)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        y_abs1, y_phase1 = fft_plot3d(gt)
        residual = abs(y_abs - y_abs1)
        plt.imsave(os.path.join(save_path, 'res' + file[:-3] + 'png'), residual)

        # plt.imsave(os.path.join(save_path,'amp_res'+file[:-3]+'png'), y_abs)
        # plt.imsave(os.path.join(save_path, 'phase' + file[:-3]+ 'png'), y_phase)

    # hazy_path = '/home/ustc-ee-huangjie/New/HighLow/Dataset/Div2K_blur/5/DIV2K_train_HR/0001.png'
    # #hazy_path = r'D:\VD\Data\SOTS\indoor\hazy\1400_10.png'
    # img_hazy = cv2.imread(hazy_path, -1)
    # img_hazy = cv2.cvtColor(img_hazy, cv2.COLOR_BGR2RGB)
    #
    # path = '/home/ustc-ee-huangjie/New/HighLow/Dataset/Div2K_noise/5/DIV2K_train_HR/0001.png'
    # #path = r'D:\VD\Data\SOTS\indoor\gt1\1436.png'
    # img = cv2.imread(path, -1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
    #
    # y1_abs, y1_phase = fft_plot3d(img_hazy)
    # y2_abs, y2_phase = fft_plot3d(img)
    #
    # ################ jiaohuan frequency
    # # y, y1, y2 = fft_exchange3d(img_clear, img, img_hazy)
    #
    #
    # plt.imsave('3m.png', y1_abs)
    # plt.imsave('3o.png', y2_abs)

    # plt.imsave('3exc.png', y)
    # plt.imsave('3exc.png', y1)

    # plt.imsave('exc.png', y2)
