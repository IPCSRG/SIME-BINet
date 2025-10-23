# -*- coding: utf-8 -*-

import cv2

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from PIL import Image
import numpy as np
import math
import time
import os
import datetime
import random
import shutil
from options.test_options import TestOptions
from models import create_model
import torchvision.transforms.functional as F
import torch
import torchvision.transforms as transforms
import lpips

from skimage.metrics import peak_signal_noise_ratio as psnr
import os


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def calculate_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


import glob
def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)
def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=opt.powerbase, method=method)))

    if convert:
        transform_list += [transforms.ToTensor()]

        if grayscale:
            if not opt.is_mask:
                transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def load_flist(flist):
    # np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
    if isinstance(flist, list):
        return flist
    # flist: image file path, image directory path, text file flist path
    if isinstance(flist, str):
        if os.path.isdir(flist):
            flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
            flist.sort()
            return flist

        if os.path.isfile(flist):
            try:
                return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
            except:
                return [flist]
    return []


def postprocess(img):
    img = (img + 1) / 2 * 255
    img = img.permute(0, 2, 3, 1)
    img = img.int().cpu().numpy().astype(np.uint8)
    return img
def postprocess1(img):
    img = img[0][0].int().cpu().numpy().astype(np.uint8)
    img = img  * 255
    return img
def sobel_func(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    ### sobel
    x = cv2.Sobel(image, cv2.CV_16S, 2, 0, ksize=3)
    y = cv2.Sobel(image, cv2.CV_16S, 0, 2, ksize=3)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Sobel = cv2.addWeighted(absX, 1, absY, 1, 0)
    image_pil = Image.fromarray(Sobel)
    # image_pil.save('./qweqw.png')
    return image_pil
# load test data

val_image = r'E:\Blind_inpainting_Datasets\DhMurals_Blind_inpainting_strokes\test\images'

best_psnr = 0
for i in range(395, 401, 5):
    if i != 0:
        pre_train_model = str(i)
        # Model and version
        opt = TestOptions().parse()  # get test options
        # hard-code some parameters for test
        opt.num_threads = 0  # test code only supports num_threads = 1
        opt.batch_size = 1  # test code only supports batch_size = 1
        opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
        opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
        # opt.epoch = str(50)
        opt.epoch = str(i)
        model = create_model(opt)  # create a model given opt.model and other options
        model.setup(opt)  # regular setup: load and print networks; create schedulers
        model.eval()
        mean_psnr = 0
        for suffix_idx in range(1):

            val_masked = r'E:\Blind_inpainting_Datasets\DhMurals_Blind_inpainting_strokes\test\input_merged_places2'
            save_dir = './results/Muals_400_trans'+'/'

            test_image_flist = load_flist(val_image)
            print(len(test_image_flist))
            test_mask_flist = load_flist(val_masked)
            print(len(test_mask_flist))


            os.makedirs(os.path.join(save_dir, 'comp'), exist_ok=True)
            os.makedirs(os.path.join(save_dir, 'org'), exist_ok=True)

            psnr_lg = []
            psnr_sk = []
            mask_num = len(test_mask_flist)
            images_list = []
            masked_list = []
            # iteration through datasets
            for idx in range(len(test_image_flist)):
                img = Image.open(test_image_flist[idx]).convert('RGB')
                masked = Image.open(test_mask_flist[idx]).convert('RGB')


                images = F.to_tensor(img) * 2 - 1.
                masksed = F.to_tensor(masked) * 2 - 1.


                images = images.unsqueeze(0)
                masksed = masksed.unsqueeze(0)


                images_list.append(images)
                masked_list.append(masksed)
            image_cat_list = []
            masked_cat_list = []
            for i in range(4,50,4):
                temp = [images_list[i-4],images_list[i-3],images_list[i-2],images_list[i-1]]
                image_cat_list.append(torch.cat(temp,dim=0))

                temp = [masked_list[i-4],masked_list[i-3],masked_list[i-2],masked_list[i-1]]
                masked_cat_list.append(torch.cat(temp,dim=0))

                if i == 48:
                    temp = [images_list[48], images_list[49], images_list[48], images_list[49]]
                    image_cat_list.append(torch.cat(temp, dim=0))

                    temp = [masked_list[48], masked_list[49], masked_list[48], masked_list[49]]
                    masked_cat_list.append(torch.cat(temp, dim=0))
                #     print(22)
                # print(i)
            for j in range(len(image_cat_list)):
                data = {'A': image_cat_list[j], 'B': masked_cat_list[j], 'C': '', 'D': masked_cat_list[j] , 'E': '' , 'A_paths': ''}
                model.set_input(data)
                with torch.no_grad():
                    model.forward()


                orig_imgs = postprocess(model.images)
                comp_imgs = postprocess(model.merged_images1)



                for get in range(4):
                    # names = test_image_flist[get+(j*4)].split('/')
                    if get+(j*4)<50:
                        # print(get,j)

                        psnr_tmp = calculate_psnr(orig_imgs[get], comp_imgs[get])
                        psnr_lg.append(psnr_tmp)
                        psnr_score = psnr(orig_imgs[get], comp_imgs[get], data_range=255)
                        psnr_sk.append(psnr_score)

                        names = test_image_flist[get+(j*4)].split('/')[-1].split('\\')
                        aaa = orig_imgs[get]
                        Image.fromarray(orig_imgs[get]).save(save_dir + '/org/' + names[-1].split('.')[0] + '.png')
                        Image.fromarray(comp_imgs[get]).save(save_dir + '/comp/' + names[-1].split('.')[0] + '.png')

            print('epoch: ' + str(opt.epoch))
            print('Finish in {}'.format(save_dir))
            print('The avg psnr is', np.mean(np.array(psnr_lg)))
            print('The avg psnr_sk is', np.mean(np.array(psnr_sk)))
            mean_psnr += np.mean(np.array(psnr_lg))
        if best_psnr < (mean_psnr):
            best_psnr = mean_psnr
        print('The all mask avg psnr is', mean_psnr, 'best: ', best_psnr)
print(best_psnr)