import torch.nn as nn
import torch.nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


class ASM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(ASM, self).__init__()
        self.conv1 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv2 = conv(3, n_feat, kernel_size, bias=bias)
        self.conv3 = nn.Sequential(
            nn.Conv2d(3, n_feat, kernel_size, padding=kernel_size // 2, stride=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2, stride=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2, stride=1, bias=False)
        )

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv1(x) + x_img
        x2 = torch.sigmoid(self.conv2(img))
        x1 = self.conv3(x1)
        x1 = x1 * x2
        x1 = x1 + x
        return x1


class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Encoder, self).__init__()

        self.encoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.encoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, x):  # x=[1/2H ,1/2W ,32]
        enc1 = self.encoder_level1(x)  # enc1 = [1/2H ,1/2W ,32]

        x = self.down12(enc1)  # x=[1/4H ,1/4W ,48]

        enc2 = self.encoder_level2(x)  # enc2=[1/4H ,1/4W ,48]

        x = self.down23(enc2)  # x=[1/8H ,1/8W ,64]

        enc3 = self.encoder_level3(x)  # enc3 = [1/8H,1/8W,64]

        return [enc1, enc2, enc3]


class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Decoder, self).__init__()

        self.decoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.decoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs  # enc1 = [H ,W ,32] enc2 = [1/2H ,1/2W ,48] enc3 = [1/4H ,1/4W ,64]

        dec3 = self.decoder_level3(enc3)  # dec3 =[1/4H,1/4W,64]

        x = self.up32(dec3, self.skip_attn2(enc2))  # x=[1/2H,1/2W,48]

        dec2 = self.decoder_level2(x)  # dec2 =[1/2H,1/2W,48]

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1, dec2, dec3]


class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):  # x=[1/4H,1/4W,64] y=[1/2H ,1/2W ,48]

        x = self.up(x)
        x = x + y
        return x


class EDSubnet(nn.Module):
    def __init__(self, in_c=3, n_feat=32, scale_unetfeats=16, kernel_size=3, reduction=4, bias=False):
        super(EDSubnet, self).__init__()

        act = nn.PReLU()
        self.shallow_feat1 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat3 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))

        self.stage_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.stage_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.asm = ASM(n_feat, kernel_size=1, bias=bias)

    def forward(self, input_img):
        H = input_img.size(2)
        W = input_img.size(3)

        # Multi-Patch Hierarchy: Split Image into four non-overlapping patches

        x2top_img = input_img[:, :, 0:int(H / 2), :]
        x2bot_img = input_img[:, :, int(H / 2):H, :]

        x1ltop_img = x2top_img[:, :, :, 0:int(W / 2)]
        x1rtop_img = x2top_img[:, :, :, int(W / 2):W]
        x1lbot_img = x2bot_img[:, :, :, 0:int(W / 2)]
        x1rbot_img = x2bot_img[:, :, :, int(W / 2):W]

        x1ltop = self.shallow_feat1(x1ltop_img)
        x1rtop = self.shallow_feat1(x1rtop_img)
        x1lbot = self.shallow_feat1(x1lbot_img)
        x1rbot = self.shallow_feat1(x1rbot_img)

        feat1_ltop = self.stage_encoder(x1ltop)
        feat1_rtop = self.stage_encoder(x1rtop)
        feat1_lbot = self.stage_encoder(x1lbot)
        feat1_rbot = self.stage_encoder(x1rbot)

        feat1_top = [torch.cat((k, v), 3) for k, v in zip(feat1_ltop, feat1_rtop)]

        feat1_bot = [torch.cat((k, v), 3) for k, v in zip(feat1_lbot, feat1_rbot)]

        feat = [torch.cat((k, v), 2) for k, v in zip(feat1_top, feat1_bot)]  # feat=[enc1,enc2,enc3]

        res = self.stage_decoder(feat)

        output = self.amm(res[0], input_img)

        return output


class AFF(nn.Module):
    def __init__(self, n_feats=32):
        super(AFF, self).__init__()

        reduction = 4
        self.glo = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feats, n_feats // reduction, 1, padding=0),
            nn.PReLU(),
            nn.BatchNorm2d(n_feats // reduction),
            nn.Conv2d(n_feats // reduction, n_feats, 1, padding=0),
            nn.BatchNorm2d(n_feats)

        )

        self.loc = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, padding=0),
            nn.BatchNorm2d(n_feats),
            nn.PReLU(),
            nn.Conv2d(n_feats, n_feats, 1, padding=0),
            nn.BatchNorm2d(n_feats),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        a = self.glo(x)
        b = self.loc(x)
        c = a + b
        y = self.sig(c)
        z = x * y

        return z


class Resblock(nn.Module):
    def __init__(self, n_feats=32):
        super(Resblock, self).__init__()
        self.PR = nn.PReLU()
        self.conv = nn.Conv2d(n_feats, n_feats, 3, padding=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, padding=1, stride=1, bias=False)

    def forward(self, x):
        res = x
        x1 = self.PR(self.conv(x))
        # cat = torch.cat([res, x1], 1)
        x2 = self.conv2(x1)
        out = self.PR(x2 + res)
        return out


class Fuse(nn.Module):

    def __init__(self, n_feats=32):
        super(Fuse, self).__init__()

        self.PR = nn.PReLU()
        self.conv1 = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, padding=1, dilation=1))
        self.conv3 = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, padding=3, dilation=3))
        self.conv5 = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, padding=5, dilation=5))
        # self.fuse = nn.Sequential(AFF(n_feats=32))
        self.fuse = nn.Sequential(eca_layer(channel=32))

    def forward(self, x):
        residual = x
        x1 = self.conv1(x)
        x2 = self.conv3(x)
        x3 = self.conv5(x)
        x4 = x1 + x2 + x3
        x5 = self.fuse(x4)
        out = residual + x5
        output = self.PR(out)
        return output


class eca_layer(nn.Module):

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class NET(nn.Module):
    def __init__(self, input_channel=32, use_GPU=True):
        super(NET, self).__init__()
        ksize = 3
        self.use_GPU = use_GPU
        self.PR = nn.PReLU()
        self.cat = nn.Conv2d(input_channel * 2, input_channel, kernel_size=3, stride=(1, 1), padding=(1, 1), bias=False)
        self.subx = EDSubnet()
        self.conv1_1 = nn.Conv2d(input_channel * 2, input_channel, 1, padding=0)
        self.fuse2 = eca_layer(channel=32)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, input_channel, ksize, padding=ksize // 2, stride=1, bias=False),
            nn.PReLU(),
        )

        self.RB1 = nn.Sequential(
            Resblock(n_feats=32),
            Resblock(n_feats=32),
            Resblock(n_feats=32),
            Fuse(n_feats=32)

        )
        self.RB2 = nn.Sequential(
            Resblock(n_feats=32),
            Resblock(n_feats=32),
            Resblock(n_feats=32),
            Fuse(n_feats=32),
        )
        self.RB3 = nn.Sequential(
            Resblock(n_feats=32),
            Resblock(n_feats=32),
            Resblock(n_feats=32),
            Fuse(n_feats=32),
        )
        self.RB4 = nn.Sequential(
            Resblock(n_feats=32),
            Resblock(n_feats=32),
            Resblock(n_feats=32),
            Fuse(n_feats=32),
        )
        self.RB5 = nn.Sequential(
            Resblock(n_feats=32),
            Resblock(n_feats=32),
            Resblock(n_feats=32),
            Fuse(n_feats=32),
        )
        self.RB6 = nn.Sequential(
            Resblock(n_feats=32),
            Resblock(n_feats=32),
            Resblock(n_feats=32),
            Fuse(n_feats=32),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(input_channel, 3, ksize, padding=ksize // 2, stride=1, bias=False)
        )

    def forward(self, input):
        x1 = self.conv1(input)

        subout = self.subx(input)

        x1 = torch.cat([x1, subout], 1)
        x1 = self.cat(x1)
        x2 = self.RB1(x1)
        x3 = self.RB2(x2)
        x4 = self.RB3(x3)
        x5 = self.RB4(x4)
        x6 = self.RB5(x5)
        x7 = self.RB6(x6)

        final = x7

        final_out = self.PR(self.fuse2(final))

        output = self.conv2(final_out)

        return output

#     def print_network(net):
#         num_params = 0
#         for param in net.parameters():
#             num_params += param.numel()
#         print(net)
#         print('Total number of parameters: %d' % num_params)
#
#
# model = NET(input_channel=32)
# print(model.print_network())
#
#
# def test():
#     net = NET(input_channel=32)
#     fms = net(Variable(torch.randn(2, 3, 32, 32)))
#     for fm in fms:
#         print(fm.size())
#
#
# test()
