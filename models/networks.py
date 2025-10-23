import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from . import attention
from .mask_model.architecture import MPN
from .mask_model.Unet import UNet
from einops import rearrange
import math
import torch.nn.functional as F
from einops.layers.torch import Rearrange

import torch
import torch.nn as nn

import math
from einops.layers.torch import Rearrange

from . import transformer

###############################################################################
# Helper Functions
###############################################################################
def PatchPositionEmbeddingSine(ksize, stride):
    temperature=10000
    feature_h = int((256-ksize)/stride)+1
    num_pos_feats = 256//2
    mask = torch.ones((feature_h, feature_h))
    y_embed = mask.cumsum(0, dtype=torch.float32)
    x_embed = mask.cumsum(1, dtype=torch.float32)
    # if self.normalize:
    #     eps = 1e-6
    #     y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
    #     x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)
    return pos
class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='xavier', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class CNNDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, norm, activ, pad_type):
        super(CNNDecoder, self).__init__()
        self.model = []
        dim = input_dim
        self.conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim, dim // 2, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)
        )
        dim //= 2
        self.conv2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim, dim // 2, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)
        )
        self.conv3 = Conv2dBlock(dim // 2, output_dim, 5, 1, 2, norm='none', activation='tanh', pad_type=pad_type)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        output = self.conv3(x2)
        return output


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', groupcount=16):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        self.norm_type = norm
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'adain_ori':
            self.norm = AdaptiveInstanceNorm2d_IN(norm_dim)
        elif norm == 'remove_render':
            self.norm = RemoveRender(norm_dim)
        elif norm == 'grp':
            self.norm = nn.GroupNorm(groupcount, norm_dim)

        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
             gpu_ids=[],bs=4):
    """Create a generator
    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Returns a generator
    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597
        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).
    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_4blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=4)
    elif netG == 'block1':
        net = Unet256(input_nc, output_nc, ngf, norm_layer=norm_layer,bs=bs,gpu_ids=gpu_ids)
    # elif netG == 'block2':
    #     net = UnetGenerator1(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    # elif netG == 'block3':
    #     net = UnetGenerator2(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    # elif netG == 'block4':
    #     net = UnetGenerator3(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    # elif netG == 'block5':
    #     net = UnetGenerator4(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    # elif netG == 'unet256':
    #     net = Unet256(input_nc, output_nc, ngf, norm_layer=norm_layer)
    elif netG == 'mask':
         net = UNet(n_channels=3, n_classes=1, bilinear=True)
         # net = MPN()
         print(111)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator
    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Returns a discriminator
    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.
        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)
        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.
    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':  # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'snpatch':  # classify if each pixel is real or fake
        net = SNDiscriminator(in_channels=input_nc, use_sigmoid=False)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        elif gan_mode == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real, is_disc=None):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'hinge':
            if is_disc:
                if target_is_real:
                    prediction = -prediction
                return self.loss(1 + prediction).mean()
            else:
                return (-prediction).mean()

        return loss


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=4,
                 padding_type='reflect', downsampling=2):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = downsampling
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out



class Unet256_cache(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(Unet256, self).__init__()
        # construct unet structure

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model1 = [nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias), ]

        model2 = [nn.LeakyReLU(0.2, True), ]
        model2 += [nn.Conv2d(ngf * 2, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model2 += [norm_layer(ngf * 2), ]

        model3 = [nn.LeakyReLU(0.2, True), ]
        model3 += [nn.Conv2d(ngf * 4, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model3 += [norm_layer(ngf * 4), ]

        model4 = [nn.LeakyReLU(0.2, True), ]
        model4 += [nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model4 += [norm_layer(ngf * 8), ]

        model5 = [nn.LeakyReLU(0.2, True), ]
        model5 += [nn.Conv2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model5 += [norm_layer(ngf * 8), ]

        model6 = [nn.LeakyReLU(0.2, True), ]
        model6 += [nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model6 += [norm_layer(ngf * 8), ]

        model7 = [nn.LeakyReLU(0.2, True), ]
        model7 += [nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model7 += [norm_layer(ngf * 8), ]

        model8 = [nn.LeakyReLU(0.2, True), ]
        model8 += [nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model8 += [nn.ReLU(True), ]
        model8 += [nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model8 += [norm_layer(ngf * 8), ]

        model9 = [nn.ReLU(True), ]
        model9 += [nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model9 += [norm_layer(ngf * 8), ]

        model10 = [nn.ReLU(True), ]
        model10 += [nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model10 += [norm_layer(ngf * 8), ]

        model11 = [nn.ReLU(True), ]
        model11 += [nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model11 += [norm_layer(ngf * 8), ]

        model12 = [nn.ReLU(True), ]
        model12 += [nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model12 += [norm_layer(ngf * 4), ]

        model13 = [nn.ReLU(True), ]
        model13 += [nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model13 += [norm_layer(ngf * 2), ]

        model14 = [nn.ReLU(True), ]
        model14 += [nn.ConvTranspose2d(ngf * 2 * 2, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model14 += [norm_layer(ngf), ]

        model15 = [nn.ReLU(True), ]
        model15 += [nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1), ]
        model15 += [nn.Tanh()]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)
        self.model9 = nn.Sequential(*model9)
        self.model10 = nn.Sequential(*model10)
        self.model11 = nn.Sequential(*model11)
        self.model12 = nn.Sequential(*model12)
        self.model13 = nn.Sequential(*model13)
        self.model14 = nn.Sequential(*model14)
        self.model15 = nn.Sequential(*model15)

        # 创建一个下采样层
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_channel1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0)
        self.conv_channel2 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.conv_channel3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.conv_channel4 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        # self.att_16 = attention.AttentionModule(512)
        # self.att_32 = attention.AttentionModule(256)
        # self.att_64 = attention.AttentionModule(128)

    def forward(self, input,image,masks):
        """Standard forward"""
        masks = masks.float()
        image_down1 = self.maxpool(image)  #128
        masks_down1 =self.maxpool(masks)
        image_masked1 = image_down1 * (1 -masks_down1) + masks_down1
        image_masked1 = self.conv_channel1(image_masked1)

        image_down2 = self.maxpool(image_masked1) #64
        masks_down2 =self.maxpool(masks_down1)
        image_masked2 = image_down2 * (1 - masks_down2) + masks_down2
        image_masked2 = self.conv_channel2(image_masked2)

        image_down3 = self.maxpool(image_masked2) #32
        masks_down3 =self.maxpool(masks_down2)
        image_masked3 = image_down3 * (1 - masks_down3) + masks_down3
        image_masked3 = self.conv_channel3(image_masked3)


        image_down4 = self.maxpool(image_masked3) #16
        masks_down4 =self.maxpool(masks_down3)
        image_masked4 = image_down4 * (1 - masks_down4) + masks_down4
        image_masked4 = self.conv_channel4(image_masked4)

        # # 使用双线性插值进行下采样，输出大小为[1, 3, 112, 112]
        # image_down1 = F.interpolate(image, size=(128, 128), mode='bilinear', align_corners=False)
        # masks_down1 = F.interpolate(masks, size=(128, 128), mode='bilinear', align_corners=False)
        #
        # image_down2 = F.interpolate(image, size=(64, 64), mode='bilinear', align_corners=False)
        # masks_down2 = F.interpolate(masks, size=(64, 64), mode='bilinear', align_corners=False)
        #
        # image_down3 = F.interpolate(image, size=(32, 32), mode='bilinear', align_corners=False)
        # masks_down3 = F.interpolate(masks, size=(32, 32), mode='bilinear', align_corners=False)
        #
        # image_down3 = F.interpolate(image, size=(32, 32), mode='bilinear', align_corners=False)
        # masks_down3 = F.interpolate(masks, size=(32, 32), mode='bilinear', align_corners=False)

        x_1 = self.model1(input)
        x_2 = self.model2(torch.cat((x_1, image_masked1), 1))
        x_3 = self.model3(torch.cat((x_2, image_masked2), 1))
        x_4 = self.model4(torch.cat((x_3, image_masked3), 1))
        x_5 = self.model5(torch.cat((x_4, image_masked4), 1))
        x_6 = self.model6(x_5)
        x_7 = self.model7(x_6)
        x_8 = self.model8(x_7)
        x_9 = self.model9(torch.cat((x_7, x_8), 1))
        x_10 = self.model10(torch.cat((x_6, x_9), 1))
        x_11 = self.model11(torch.cat((x_5, x_10), 1))
        # x_11 = self.att_16(x_11)
        x_12 = self.model12(torch.cat((x_4, x_11), 1))
        # x_12 = self.att_32(x_12)
        x_13 = self.model13(torch.cat((x_3, x_12), 1))
        # x_13 = self.att_64(x_13)
        x_14 = self.model14(torch.cat((x_2, x_13), 1))
        output = self.model15(torch.cat((x_1, x_14), 1))

        return output

class Unet256(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d,bs=4,gpu_ids=[]):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(Unet256, self).__init__()
        # construct unet structure

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        dim = 256

        self.patch_to_embedding_masked_image = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=4, p2=4),
            nn.Linear(4 * 4 * 3, dim)
        )

        self.patch_to_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=4, p2=4),
            nn.Linear(4 * 4 * 3, dim)
        )
        self.transformer_enc = transformer.TransformerEncoders(dim, nhead=2, num_encoder_layers=9,
                                                               dim_feedforward=dim * 2, activation='gelu')
        self.cnn_dec = CNNDecoder(256, 3, 'ln', 'lrelu', 'reflect')


        input_pos = PatchPositionEmbeddingSine(ksize=4, stride=4)
        self.input_pos = input_pos.unsqueeze(0).repeat(bs, 1, 1, 1).to(gpu_ids[0])
        self.input_pos = self.input_pos.flatten(2).permute(2, 0, 1)


        #
        # # 创建一个下采样层
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv_channel1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0)
        # self.conv_channel2 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        # self.conv_channel3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        # self.conv_channel4 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        # # self.att_16 = attention.AttentionModule(512)
        # # self.att_32 = attention.AttentionModule(256)
        # # self.att_64 = attention.AttentionModule(128)

    def forward(self, input,image_masked):
        """Standard forward"""

        image_masked_patch_embedding = self.patch_to_embedding_masked_image(image_masked)

        patch_embedding = self.patch_to_embedding(input)
        content = self.transformer_enc(patch_embedding.permute(1, 0, 2),image_masked_patch_embedding.permute(1, 0, 2), src_pos=self.input_pos)
        bs, L, C  = patch_embedding.size()
        content = content.permute(1,2,0).view(bs, C, int(math.sqrt(L)), int(math.sqrt(L)))
        output = self.cnn_dec(content)

        return output


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


# Ref: PENNet
from .spectral_norm import use_spectral_norm


class SNDiscriminator(nn.Module):
    def __init__(self, in_channels=3, use_sigmoid=False, use_sn=True, init_weights=True):
        super(SNDiscriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        cnum = 64
        self.encoder = nn.Sequential(
            use_spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=cnum,
                                        kernel_size=5, stride=2, padding=1, bias=False), use_sn=use_sn),
            nn.LeakyReLU(0.2, inplace=True),

            use_spectral_norm(nn.Conv2d(in_channels=cnum, out_channels=cnum * 2,
                                        kernel_size=5, stride=2, padding=1, bias=False), use_sn=use_sn),
            nn.LeakyReLU(0.2, inplace=True),

            use_spectral_norm(nn.Conv2d(in_channels=cnum * 2, out_channels=cnum * 4,
                                        kernel_size=5, stride=2, padding=1, bias=False), use_sn=use_sn),
            nn.LeakyReLU(0.2, inplace=True),

            use_spectral_norm(nn.Conv2d(in_channels=cnum * 4, out_channels=cnum * 8,
                                        kernel_size=5, stride=1, padding=1, bias=False), use_sn=use_sn),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.classifier = nn.Conv2d(in_channels=cnum * 8, out_channels=1, kernel_size=5, stride=1, padding=1)

    def forward(self, x):
        x = self.encoder(x)
        label_x = self.classifier(x)
        if self.use_sigmoid:
            label_x = torch.sigmoid(label_x)
        return label_x