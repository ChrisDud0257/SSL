from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
import torch
import torch.optim as optim

from basicsr.utils.registry import ARCH_REGISTRY

class OrthorTransform(nn.Module):
    def __init__(self, c_dim, feat_hw, groups): #feat_hw: width or height (let width == heigt)
        super(OrthorTransform, self).__init__()

        self.groups = groups
        self.c_dim = c_dim
        self.feat_hw = feat_hw
        self.weight = nn.Parameter(torch.randn(1, feat_hw, c_dim))
        self.opt_orth = optim.Adam(self.parameters(), lr=1e-3, betas=(0.5, 0.99))
    def forward(self, feat):
        pred = feat * self.weight.expand_as(feat)
        return pred, self.weight.view(self.groups, -1)

class CodeReduction(nn.Module):
    def __init__(self, c_dim, feat_hw, blocks = 4, prob=False):
        super(CodeReduction, self).__init__()
        self.body = nn.Sequential(
            nn.Linear(c_dim, c_dim*blocks),
            nn.LeakyReLU(0.2, True)
        )
        self.trans = OrthorTransform(c_dim=c_dim*blocks, feat_hw=feat_hw, groups = blocks)
        self.leakyrelu = nn.LeakyReLU(0.2, True)
    def forward(self, feat):
        feat = self.body(feat)
        feat, weight = self.trans(feat)
        feat = self.leakyrelu(feat)
        return feat, weight


@ARCH_REGISTRY.register()
class MOD(nn.Module):
    '''
        output is not normalized
    '''

    def __init__(self, num_in_ch, num_feat, num_expert=12):
        super(MOD, self).__init__()
        self.num_expert = num_expert
        self.num_feat = num_feat

        self.FE = nn.Sequential(
            nn.Conv2d(3, num_feat, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat, num_feat, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_feat),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat, num_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_feat * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat * 2, num_feat * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_feat * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat * 2, num_feat * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_feat * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat * 4, num_feat * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_feat * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat * 4, num_feat * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_feat * 4),
            nn.LeakyReLU(0.2, True),
        )

        self.w_gating1 = nn.Parameter(torch.randn(num_feat * 4, self.num_expert))

        m_classifier = [
            nn.Linear(num_feat * 4, num_feat // 2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(num_feat // 2, 1)
        ]
        self.classifier = nn.Sequential(*m_classifier)
        self.classifiers = nn.ModuleList()
        for _ in range(self.num_expert):
            self.classifiers.append(self.classifier)

        self.orthonet = CodeReduction(c_dim=num_feat * 4, feat_hw=1, blocks=self.num_expert)

    def forward(self, x, routing=None):
        feature = self.FE(x)
        B, C, H, W = feature.shape
        feature = feature.view(B, -1, H * W).permute(0, 2, 1)
        if routing == None:
            routing = torch.einsum('bnd,de->bne', feature, self.w_gating1)
            routing = routing.softmax(dim=-1)

        feature, ortho_weight = self.orthonet(feature)
        feature = torch.split(feature, [feature.shape[-1] // self.num_expert] * self.num_expert, dim=-1)

        # soft routing
        # output =  self.classifiers[0](feature[0]) * routing[:,:,[0]]
        # for i in range(1, self.num_expert1):
        #     output = output + self.classifiers[i](feature[i]) * routing[:,:,[i]]

        # hard routing
        routing_top = torch.max(routing, dim=-1)[1].unsqueeze(-1).float()
        for i in range(self.num_expert):
            if i == 0:
                output = self.classifiers[0](feature[0])
            else:
                output = torch.where(routing_top == i, self.classifiers[i](feature[i]), output)
        return output, routing, feature, ortho_weight


@ARCH_REGISTRY.register()
class VGGStyleDiscriminator(nn.Module):
    """VGG style discriminator with input size 128 x 128 or 256 x 256.

    It is used to train SRGAN, ESRGAN, and VideoGAN.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features.Default: 64.
    """

    def __init__(self, num_in_ch, num_feat, input_size=128):
        super(VGGStyleDiscriminator, self).__init__()
        self.input_size = input_size
        assert self.input_size == 128 or self.input_size == 256, (
            f'input size must be 128 or 256, but received {input_size}')

        self.conv0_0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_feat, affine=True)

        self.conv1_0 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv2d(num_feat * 2, num_feat * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(num_feat * 2, affine=True)

        self.conv2_0 = nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(num_feat * 4, affine=True)
        self.conv2_1 = nn.Conv2d(num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(num_feat * 4, affine=True)

        self.conv3_0 = nn.Conv2d(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv3_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.conv4_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv4_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        if self.input_size == 256:
            self.conv5_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
            self.bn5_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
            self.conv5_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
            self.bn5_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.linear1 = nn.Linear(num_feat * 8 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        assert x.size(2) == self.input_size, (f'Input size must be identical to input_size, but received {x.size()}.')

        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))  # output spatial size: /2

        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))  # output spatial size: /4

        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(self.conv2_1(feat)))  # output spatial size: /8

        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))  # output spatial size: /16

        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))  # output spatial size: /32

        if self.input_size == 256:
            feat = self.lrelu(self.bn5_0(self.conv5_0(feat)))
            feat = self.lrelu(self.bn5_1(self.conv5_1(feat)))  # output spatial size: / 64

        # spatial size: (4, 4)
        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out

@ARCH_REGISTRY.register()
class Discriminator_VGG_192(nn.Module):
    def __init__(self, in_chl = 3, nf = 64):
        super(Discriminator_VGG_192, self).__init__()
        # in: [in_chl, 192, 192]
        self.conv0_0 = nn.Conv2d(in_chl, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(nf, affine=True)
        # [nf, 96, 96]
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
        # [nf*2, 48, 48]
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)
        # [nf*4, 24, 24]
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(nf * 8, affine=True)
        # [nf*8, 12, 12]
        self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(nf * 8, affine=True)
        # [nf*8, 6, 6]
        self.conv5_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn5_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv5_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn5_1 = nn.BatchNorm2d(nf * 8, affine=True)
        # [nf*8, 3, 3]
        self.linear1 = nn.Linear(nf * 8 * 3 * 3, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))

        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))

        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))

        fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))

        fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
        fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))

        fea = self.lrelu(self.bn5_0(self.conv5_0(fea)))
        fea = self.lrelu(self.bn5_1(self.conv5_1(fea)))

        fea = fea.view(fea.size(0), -1)
        fea = self.lrelu(self.linear1(fea))
        out = self.linear2(fea)
        return out

@ARCH_REGISTRY.register()
class DiscriminatorSN_VGG_192(nn.Module):
    def __init__(self, in_chl = 3, nf = 64):
        super(DiscriminatorSN_VGG_192, self).__init__()
        norm = spectral_norm

        # in: [in_chl, 192, 192]
        self.conv0_0 = nn.Conv2d(in_chl, nf, 3, 1, 1, bias=True)
        self.conv0_1 = norm(nn.Conv2d(nf, nf, 4, 2, 1, bias=False))

        # [nf, 96, 96]
        self.conv1_0 = norm(nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False))

        self.conv1_1 = norm(nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False))

        # [nf*2, 48, 48]
        self.conv2_0 = norm(nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False))

        self.conv2_1 = norm(nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False))

        # [nf*4, 24, 24]
        self.conv3_0 = norm(nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False))

        self.conv3_1 = norm(nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False))

        # [nf*8, 12, 12]
        self.conv4_0 = norm(nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False))

        self.conv4_1 = norm(nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False))

        # [nf*8, 6, 6]
        self.conv5_0 = norm(nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False))

        self.conv5_1 = norm(nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False))

        # [nf*8, 3, 3]
        self.linear1 = nn.Linear(nf * 8 * 3 * 3, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


    def forward(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.conv0_1(fea))

        fea = self.lrelu(self.conv1_0(fea))
        fea = self.lrelu(self.conv1_1(fea))

        fea = self.lrelu(self.conv2_0(fea))
        fea = self.lrelu(self.conv2_1(fea))

        fea = self.lrelu(self.conv3_0(fea))
        fea = self.lrelu(self.conv3_1(fea))

        fea = self.lrelu(self.conv4_0(fea))
        fea = self.lrelu(self.conv4_1(fea))

        fea = self.lrelu(self.conv5_0(fea))
        fea = self.lrelu(self.conv5_1(fea))

        fea = fea.view(fea.size(0), -1)
        fea = self.lrelu(self.linear1(fea))
        out = self.linear2(fea)
        return out


@ARCH_REGISTRY.register(suffix='basicsr')
class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out

@ARCH_REGISTRY.register()
class UNetDiscriminatorSNv1(nn.Module):
    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSNv1, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))

        # upsample

        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions

        self.conv3 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv4 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)  #b,hum_feat,h,w
        _, _, h0, w0=x0.shape
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)  #b,hum_feat,h//2,w//2
        # _, _, h1, w1 = x1.shape

        # upsample
        x1 = F.interpolate(x1, size=(h0, w0), mode='bilinear', align_corners=False)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x2 = x2 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)
        out = self.conv4(out)


        return out

if __name__ == '__main__':
    input = torch.randn(12,1,25,25)
    discriminator = UNetDiscriminatorSNv1(num_in_ch=1, num_feat=64, skip_connection=True)
    output = discriminator(input)
    print(output.shape)