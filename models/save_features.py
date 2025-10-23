

import matplotlib
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import torch
import math
from PIL import Image
import numpy as np
from matplotlib import _api, cbook, cm



def get_features(features,feature_name):
    aa = features[0, :].unsqueeze(0)
    bb = features[1, :].unsqueeze(0)
    cc = features[2, :].unsqueeze(0)
    dd = features[3, :].unsqueeze(0)

    for j, feature in enumerate([aa,bb,cc,dd]):
        feature_map = feature
        batch, channels, height, width = feature_map.shape

        preprocess = transforms.Resize([height, width])
        feature_map = preprocess(feature_map)

        blocks = torch.chunk(feature_map[0].cpu(), channels, dim=0)
        # n = min(width, channels)  # number of plots
        n = channels
        if n == 3:
            fig, ax = plt.subplots(math.ceil(3), 1, tight_layout=True)  # 8 rows x n/8 cols
        else:
            if n == 128:
                fig, ax = plt.subplots(math.ceil(n / 16), 16, tight_layout=True)
            elif n == 256:
                fig, ax = plt.subplots(math.ceil(n / 16), 16, tight_layout=True)
            elif n == 512 :
                fig, ax = plt.subplots(math.ceil(n / 24), 24, tight_layout=True)
            else:
                fig, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 8 rows x n/8 cols
        ax = ax.ravel()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        for i in range(n):
            aa = blocks[i].squeeze().detach().numpy()
            plt.imsave('./result_features_out/' + feature_name + '_' + str(j) + '_' + str(width) + '_' + str(i) +'.png',aa)
            ax[i].imshow(blocks[i].squeeze().detach().numpy())  # cmap='gray'
            ax[i].axis('off')
        plt.savefig('./result_features_out/' + feature_name + '_' + str(j) + '_' + str(width) + '.png', dpi=width * (n / 8) + 100, bbox_inches='tight')
        # plt.savefig('./G5_up_' + str(width) + '.jpg', dpi=128 * (n / 8) + 100, bbox_inches='tight')
        # plt.savefig('./G1_only_'+str(width)+'.jpg', dpi=128*(n / 8)+100, bbox_inches='tight')
        # plt.savefig('./G1_only_up_' + str(width) + '.jpg', dpi=128 * (n / 8) + 100, bbox_inches='tight')
        plt.show()
        plt.close()