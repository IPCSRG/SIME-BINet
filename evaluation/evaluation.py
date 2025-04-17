import os
import glob
import shutil
import lpips
import numpy as np
import argparse
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from dataloader.image_folder import make_dataset
from util import util
import torch
import torch.nn.functional as F

lpips_alex = lpips.LPIPS(net='alex')


def calculate_score(img_gt, img_test):
    """
    function to calculate the image quality score
    :param img_gt: original image
    :param img_test: generated image
    :return: mae, ssim, psnr
    """


    psnr_score = psnr(img_gt, img_test, data_range=1)

    ssim_score = ssim(img_gt, img_test, multichannel=True, data_range=1, win_size=11)

    lpips_dis = lpips_alex(torch.from_numpy(img_gt).permute(2, 0, 1), torch.from_numpy(img_test).permute(2, 0, 1), normalize=True)

    return  ssim_score, psnr_score, lpips_dis.data.numpy().item()



def evl(test_path = '',org_path = ''):
    parser = argparse.ArgumentParser(description='Image quality evaluations on the dataset')

    parser.add_argument('--gt_path', type=str, default=org_path, help='path to original gt data')
    parser.add_argument('--g_path', type=str, default=test_path, help='path to the generated data')
    parser.add_argument('--save_path', type=str, default=None, help='path to save the best results')
    parser.add_argument('--center', action='store_true',
                        help='only calculate the center masked regions for the image quality')
    parser.add_argument('--num_test', type=int, default=0, help='how many examples to load for testing')

    args = parser.parse_args()
    gt_paths, gt_size = make_dataset(args.gt_path)
    g_paths, g_size = make_dataset(args.g_path)
    print(args.g_path)

    ssims = []
    psnrs = []
    lpipses = []

    size = args.num_test if args.num_test > 0 else gt_size

    for i in range(size):
        gt_img = Image.open(gt_paths[i]).resize([256, 256]).convert('RGB')
        gt_numpy = np.array(gt_img).astype(np.float32) / 255.0

        g_img = Image.open(g_paths[i]).resize([256, 256]).convert('RGB')
        g_numpy = np.array(g_img).astype(np.float32) / 255.0

        ssim_score, psnr_score, lpips_score = calculate_score(gt_numpy, g_numpy)


        ssims.append(ssim_score)
        psnrs.append(psnr_score)
        lpipses.append(lpips_score)

    print('{:>10},{:>10},{:>10}'.format( 'SSIM', 'PSNR', 'LPIPS'))
    print('{:10.4f},{:10.4f},{:10.4f}'.format( np.mean(ssims), np.mean(psnrs),
                                                       np.mean(lpipses)))
    print('{:10.4f},{:10.4f},{:10.4f}'.format( np.var(ssims), np.var(psnrs), np.var(lpipses)))


def evl_single(test_path = '',org_path = ''):
    parser = argparse.ArgumentParser(description='Image quality evaluations on the dataset')

    parser.add_argument('--gt_path', type=str, default=org_path, help='path to original gt data')
    parser.add_argument('--g_path', type=str, default=test_path, help='path to the generated data')
    parser.add_argument('--save_path', type=str, default=None, help='path to save the best results')
    parser.add_argument('--center', action='store_true',
                        help='only calculate the center masked regions for the image quality')
    parser.add_argument('--num_test', type=int, default=0, help='how many examples to load for testing')

    args = parser.parse_args()
    gt_paths = args.gt_path
    g_paths = args.g_path
    print(args.g_path)

    ssims = []
    psnrs = []
    lpipses = []

    gt_img = Image.open(gt_paths).resize([256, 256]).convert('RGB')
    gt_numpy = np.array(gt_img).astype(np.float32) / 255.0

    g_img = Image.open(g_paths).resize([256, 256]).convert('RGB')
    g_numpy = np.array(g_img).astype(np.float32) / 255.0

    ssim_score, psnr_score, lpips_score = calculate_score(gt_numpy, g_numpy)


    ssims.append(ssim_score)
    psnrs.append(psnr_score)
    lpipses.append(lpips_score)

    print('{:>10},{:>10},{:>10}'.format( 'SSIM', 'PSNR', 'LPIPS'))
    print('{:10.4f},{:10.4f},{:10.4f}'.format( np.mean(ssims), np.mean(psnrs),
                                                       np.mean(lpipses)))
    print('{:10.4f},{:10.4f},{:10.4f}'.format(np.var(ssims), np.var(psnrs), np.var(lpipses)))
if __name__ == '__main__':

    org_path_list = [
        './test/images/'
                     ]

    comp_path_list = [
        './01 TransCNN-HAE-1.0/',
        './02 Blind_Omni_Wav_Net/',
        './03 Decontamination_Transformer/',
        './04 comp/'
    ]


    for num in range(0,1):
        org_path_t = org_path_list[num]
        path = comp_path_list[3]
        image_list = os.listdir(path)
        for i in image_list:

            COMP_PATH = path + i
            ORG_PATH = org_path_t + i
            evl_single(test_path=COMP_PATH, org_path=ORG_PATH)



