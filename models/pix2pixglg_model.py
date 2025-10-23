import torch
from .base_model import BaseModel
from . import networks
from .loss import l1_loss_mask, VGG16FeatureExtractor, style_loss, perceptual_loss, TV_loss
import torch.nn as nn
from .utils.dice_score import dice_loss
import torch.nn.functional as F
import numpy as np
# global + local + global network

import math
import scipy.stats as st






class Pix2PixGLGModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', netG='unet_256', preprocess='resize', no_dropout=True, load_size=256,
                            is_mask=True, gan_mode='nogan')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        self.loss_names = ['G', 'G_content', 'G_style', 'G_tv', 'G_seg']
        if self.opt.gan_mode != 'nogan':
            self.loss_names += ['D_real', 'D_fake', 'G_GAN']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['masked_images1',  'images',  'seg1','masks','images_masked','merged_images1','output_images1' ,'sigmoid_seg1']

        self.model_names = ['G1', 'Mask1']
        if self.opt.gan_mode != 'nogan':
            self.model_names += ['D']
        # define networks (both generator and discriminator)
        self.netMask1 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'mask', opt.norm,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, gpu_ids=self.gpu_ids,bs=opt.batch_size)
        self.netG1 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'block1', opt.norm,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, gpu_ids=self.gpu_ids,bs=opt.batch_size)



        if self.isTrain:  # define a discriminator;
            if opt.gan_mode != 'nogan':
                self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                              opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterion_mask = nn.MSELoss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_Mask1 = torch.optim.Adam(self.netMask1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G1 = torch.optim.Adam(self.netG1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


            if opt.gan_mode != 'nogan':
                self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_Mask1)
            self.optimizers.append(self.optimizer_G1)


            self.lossNet = VGG16FeatureExtractor()

            self.segcriterion = nn.BCEWithLogitsLoss()
            self.lossNet.cuda(opt.gpu_ids[0])

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.images = input['A' ].to(self.device)
        self.masks = input['B' ].to(self.device)
        self.maskedimage_places2 = input['D' ].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.isTrain:
            ### train
            self.images_masked = self.images * (1 - self.masks) + self.maskedimage_places2 * self.masks
            self.seg1 = self.netMask1(self.images_masked)
            self.sigmoid_seg1 = (self.seg1 > 0.5).int()
            self.masked_images1 = self.images_masked * (1 - self.sigmoid_seg1) + self.sigmoid_seg1
            self.output_images1 = self.netG1(self.images_masked,self.masked_images1)
            self.merged_images1 = self.images_masked * (1 - self.sigmoid_seg1) + self.output_images1 * self.sigmoid_seg1
        else:
            ### test
            self.images_masked = self.maskedimage_places2
            self.seg1 = self.netMask1(self.images_masked)
            self.sigmoid_seg1 = (self.seg1 > 0.5).int()
            self.masked_images1 = self.images_masked * (1 - self.sigmoid_seg1) + self.sigmoid_seg1
            self.output_images1 = self.netG1(self.images_masked,self.masked_images1)
            self.merged_images1 = self.images_masked * (1 - self.sigmoid_seg1) + self.output_images1 * self.sigmoid_seg1


    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(self.merged_images1.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False, is_disc=True)
        # Real
        pred_real = self.netD(self.images)
        self.loss_D_real = self.criterionGAN(pred_real, True, is_disc=True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""

        self.loss_G_seg = self.segcriterion(self.seg1, self.masks)
        self.loss_G_seg += dice_loss(self.seg1.squeeze(1), self.masks.squeeze(1), multiclass=False)

        loss_G_valid = l1_loss_mask(self.output_images1 * (1 - self.masks), self.images * (1 - self.masks),
                                    (1 - self.masks))
        loss_G_hole = l1_loss_mask(self.output_images1 * self.masks, self.images * self.masks, self.masks)
        self.loss_G = loss_G_valid + 6 * loss_G_hole


        real_B_feats = self.lossNet(self.images)
        fake_B_feats = self.lossNet(self.output_images1)
        comp_B_feats = self.lossNet(self.merged_images1)

        self.loss_G_tv = TV_loss(self.merged_images1 * self.masks)
        self.loss_G_style = style_loss(real_B_feats, fake_B_feats) + style_loss(real_B_feats, comp_B_feats)
        self.loss_G_content = perceptual_loss(real_B_feats, fake_B_feats) + perceptual_loss(real_B_feats,
                                                                                            comp_B_feats)
        self.loss_G = self.loss_G_seg + self.loss_G + 0.05 * self.loss_G_content + 120 * self.loss_G_style + 0.1 * self.loss_G_tv

        # G(A) should fake the discriminator

        if self.opt.gan_mode != 'nogan':
            pred_fake = self.netD(self.merged_images1)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True, is_disc=False)
            self.loss_G = self.loss_G + 0.1 * self.loss_G_GAN
        # self.loss_G = self.loss_G_seg
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)

        if self.opt.gan_mode != 'nogan':
            # update D
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()  # set D's gradients to zero
            self.backward_D()  # calculate gradients for D
            self.optimizer_D.step()  # update D's weights

            # update G
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G

        # update G
        self.optimizer_Mask1.zero_grad()
        self.optimizer_G1.zero_grad()  # set G's gradients to zero

        self.backward_G()  # calculate graidents for G
        self.optimizer_Mask1.step()
        self.optimizer_G1.step()  # udpate G's weights
