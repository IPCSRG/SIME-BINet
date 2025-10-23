# SIME-BINet

code for paper "Self-information and prediction mask enhanced blind inpainting network for dunhuang mural images".  Simple and efficient implementation.

## Prerequisites

- Python 3.7
- CUDA 11.7  
- PyTorch 1.13.1

## TODO

- [x] Releasing dataset.
- [x] Releasing evaluation code.
- [x] Releasing inference codes.
- [x] Releasing pre-trained weights.
- [x] Releasing training codes.


## Download Datasets

We use [DhMurals1714](https://github.com/qinnzou/mural-image-inpainting) datasets. And the dataset containing the masks and places2 images used in the paper can be downloaded [here](https://drive.google.com/file/d/1Qdb2webAJqgeweYm-ZwYCucyLoBwSebH/view?usp=drive_link).

## Run
1. train the model
```
train.py --dataroot no_use --name Muals_400_trans --model pix2pixglg --netG1 unet_256 --netD snpatch --gan_mode lsgan --input_nc 4 --no_dropout --direction AtoB --display_id 0
```
2. test the model
```
test_and_save_epoch_masked_TRANS.py --dataroot no_use --name Muals_400_trans --model pix2pixglg --netG1 unet_256 --gan_mode nogan --input_nc 4 --no_dropout --direction AtoB --gpu_ids 0
```
## pre-trained weights

It's fair to suggest that retrain with your own dataset. You can also use the weight file directly.
[Mural](https://drive.google.com/file/d/1VewFYU5W_AhyH0qw6d6FQj0OOFxKM6pR/view?usp=sharing)

## Citation
```
@article{MENG2025111769,
title = {Self-information and prediction mask enhanced blind inpainting network for dunhuang murals},
journal = {Engineering Applications of Artificial Intelligence},
volume = {159},
pages = {111769},
year = {2025},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2025.111769},
}
```
## Acknowledgments

Please consider to cite their papers, if used the mural dataset.
This code based on [LGNet](https://github.com/weizequan/LGNet). Please consider to cite their papers.
```
@ARTICLE{9730792,
  author={Quan, Weize and Zhang, Ruisong and Zhang, Yong and Li, Zhifeng and Wang, Jue and Yan, Dong-Ming},
  journal={IEEE Transactions on Image Processing}, 
  title={Image Inpainting With Local and Global Refinement}, 
  year={2022},
  volume={31},
  pages={2405-2420}
}
@article{muralnet2022,
  title={Line Drawing Guided Progressive Inpainting of Mural Damages},
  author={Luxi Li and Qin Zou and Fan Zhang and Hongkai Yu and Long Chen and Chengfang Song and Xianfeng Huang and Xiaoguang Wang},
  journal={ArXiv 2211.06649},
  pages={1--12},
  year={2022},
}
```
