import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

########### Multi feat segmentation head ###########

def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))

class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            # self.bn = norm_layer(out_planes, eps=bn_eps)
            self.bn = nn.GroupNorm(16, out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


class Attention(nn.Module):
    def __init__(self, in_places, scale=8, eps=1e-6):
        super(Attention, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.in_places = in_places
        self.l2_norm = l2_norm
        self.eps = eps

        self.query_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = self.l2_norm(Q).permute(-3, -1, -2)
        K = self.l2_norm(K)

        tailor_sum = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps))
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, width * height)

        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)

        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (self.gamma * weight_value).contiguous()


class AttAgreg(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convblk = ConvBnRelu(in_ch, out_ch, ksize=1, stride=1, pad=0)
        self.conv_atten = Attention(out_ch)

    def forward(self, f1, f2, f3):
        fcat = torch.cat([f1,f2,f3], dim=1)
        feat = self.convblk(fcat)
        atten = self.conv_atten(feat)
        feat_out = atten + feat

        return feat_out


class conv_gn_layer(nn.Module):
    def __init__(self,in_ch,out_ch,upsample=False)-> None:
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (3, 3),stride=1, padding=1, bias=False),
            nn.GroupNorm(16, out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x

class conv_3(nn.Module):
    def __init__(self,in_ch,out_ch,n_upsample=0) -> None:
        super().__init__()
        blocks = [conv_gn_layer(in_ch,out_ch,upsample=bool(n_upsample))]
        if n_upsample > 1:
            for _ in range(1, n_upsample):
                blocks.append(conv_gn_layer(out_ch, out_ch,upsample=True))
        self.block = nn.Sequential(*blocks)
    def forward(self,x):
        return self.block(x)


class multi_feat_head(nn.Module):
    def __init__(self,
                 sz_fmap:int,
                 n_upsample:int,
                 n_class:int,
                 n_fmap= 64,
                 dropout=0.2) -> None:

        super().__init__()
        # Extract segmentation features from the ViT
        self.layers_feat1 = nn.Sequential(
                    nn.Conv2d(sz_fmap, n_fmap, kernel_size=(1, 1)),
                    conv_3(n_fmap,n_fmap,n_upsample=n_upsample)
                    )
        self.layers_feat2 = nn.Sequential(
                    nn.Conv2d(sz_fmap, n_fmap, kernel_size=(1, 1)),
                    conv_3(n_fmap,n_fmap,n_upsample=n_upsample)
                    )
        self.layers_feat3 = nn.Sequential(
                    nn.Conv2d(sz_fmap, n_fmap, kernel_size=(1, 1)),
                    conv_3(n_fmap,n_fmap,n_upsample=n_upsample)
                    )
        # Aggregate the different features
        self.attention = AttAgreg(n_fmap * 3, n_fmap * 3)
        # self.final_conv = nn.Sequential(
        #             conv_gn_layer(n_fmap * 3,n_fmap),
        #             nn.Conv2d(n_fmap, n_class, kernel_size=1, padding=0)
        #             )
        self.final_conv = nn.Conv2d(n_fmap * 3, n_class, kernel_size=1, padding=0)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)


    def forward(self, feat1, feat2, feat3):

        x1 = self.layers_feat1(feat1)
        x2 = self.layers_feat2(feat2)
        x3 = self.layers_feat3(feat3)

        out = self.dropout(self.attention(x1,x2,x3))
        out = self.final_conv(out)
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)

        return out

