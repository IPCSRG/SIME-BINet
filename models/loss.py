import torch
from torchvision import models
import torch.nn as nn

def l1_loss_mask(inputs, targets, mask):
    loss = torch.sum(torch.abs(inputs - targets), dim=[1,2,3])
    xs = targets.size()
    ratio = torch.sum(mask, dim=[1,2,3]) * xs[1]  # mask: BWH1. if mask = BWH3, then remove '*3'
    loss_mean = torch.mean(loss / (ratio + 1e-12))  # avoid mask = 0
    return loss_mean

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

def style_loss(A_feats, B_feats):
    assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
    loss_value = 0.0
    for i in range(len(A_feats)):
        A_feat = A_feats[i]
        B_feat = B_feats[i]
        _, c, w, h = A_feat.size()
        A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
        B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
        A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
        B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
        loss_value += torch.mean(torch.abs(A_style - B_style)/(c * w * h))
    return loss_value

def TV_loss(x):
    h_x = x.size(2)
    w_x = x.size(3)
    h_tv = torch.mean(torch.abs(x[:,:,1:,:]-x[:,:,:h_x-1,:]))
    w_tv = torch.mean(torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1]))
    return h_tv + w_tv

def perceptual_loss(A_feats, B_feats):
    assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
    loss_value = 0.0
    for i in range(len(A_feats)):
        A_feat = A_feats[i]
        B_feat = B_feats[i]
        loss_value += torch.mean(torch.abs(A_feat - B_feat))
    return loss_value

####################################################################################################
# trained LPIPS loss
####################################################################################################
def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x/(norm_factor+eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class LPIPSLoss(nn.Module):
    """
    Learned perceptual metric
    https://github.com/richzhang/PerceptualSimilarity
    """
    def __init__(self, use_dropout=True, ckpt_path=None):
        super(LPIPSLoss, self).__init__()
        self.path = ckpt_path
        self.net = VGG16()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self):
        self.load_state_dict(torch.load(self.path, map_location=torch.device("cpu")), strict=False)
        print("loaded pretrained LPIPS loss from {}".format(self.path))

    def _get_features(self, vgg_f):
        names = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']
        feats = []
        for i in range(len(names)):
            name = names[i]
            feat = vgg_f[name]
            feats.append(feat)
        return feats

    def forward(self, x, y):
        x_vgg, y_vgg = self._get_features(self.net(x)), self._get_features(self.net(y))
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        reses = []
        loss = 0

        for i in range(len(self.chns)):
            x_feats, y_feats = normalize_tensor(x_vgg[i]), normalize_tensor(y_vgg[i])
            diffs = (x_feats - y_feats) ** 2
            res = spatial_average(lins[i].model(diffs))
            loss += res
            reses.append(res)

        return loss

#
# class PerceptualLoss(nn.Module):
#     r"""
#     Perceptual loss, VGG-based
#     https://arxiv.org/abs/1603.08155
#     https://github.com/dxyang/StyleTransfer/blob/master/utils.py
#     """
#
#     def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 0.0]):
#         super(PerceptualLoss, self).__init__()
#         self.add_module('vgg', VGG16())
#         self.criterion = nn.L1Loss()
#         self.weights = weights
#
#     def __call__(self, x, y):
#         # Compute features
#         x_vgg, y_vgg = self.vgg(x), self.vgg(y)
#
#         content_loss = 0.0
#         content_loss += self.weights[0] * self.criterion(x_vgg['relu1_2'], y_vgg['relu1_2']) if self.weights[0] > 0 else 0
#         content_loss += self.weights[1] * self.criterion(x_vgg['relu2_2'], y_vgg['relu2_2']) if self.weights[1] > 0 else 0
#         content_loss += self.weights[2] * self.criterion(x_vgg['relu3_3'], y_vgg['relu3_3']) if self.weights[2] > 0 else 0
#         content_loss += self.weights[3] * self.criterion(x_vgg['relu4_3'], y_vgg['relu4_3']) if self.weights[3] > 0 else 0
#         content_loss += self.weights[4] * self.criterion(x_vgg['relu5_3'], y_vgg['relu5_3']) if self.weights[4] > 0 else 0
#
#         return content_loss


class Normalization(nn.Module):
    def __init__(self, device):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(23, 26):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(26, 28):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(28, 30):
            self.relu5_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
           param.requires_grad = False

    def forward(self, x,):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)

        relu4_1 = self.relu4_1(relu3_3)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)

        relu5_1 = self.relu5_1(relu4_3)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
        }
        return out