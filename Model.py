import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch import einsum
import time

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class Attention(nn.Module):
    def __init__(self, dim, length):
        super().__init__()
        self.pc_proj_q = nn.Linear(dim, 1, bias=False)
        self.bias_pc_proj_q = nn.Parameter(torch.FloatTensor([1.]))
        self.pc_proj_k = nn.Linear(dim, 1, bias=False)
        self.bias_pc_proj_k = nn.Parameter(torch.FloatTensor([1.]))
        self.mlp1 = nn.Sequential(
            nn.Linear(length, 1, bias=False),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(length, length, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(length, 1, bias=False),
        )

    def forward(self, q, k):
        Sigma_q = self.pc_proj_q(q) + self.bias_pc_proj_q
        Sigma_k = self.pc_proj_k(k) + self.bias_pc_proj_k
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        Sigma = einsum('b h i d, b h j d -> b h i j', Sigma_q, Sigma_k)

        diag_sim = torch.diagonal(sim, dim1=-2, dim2=-1)
        sim_norm = sim - torch.diag_embed(diag_sim)
        theta = self.mlp1(sim_norm).squeeze(-1)
        theta = self.mlp2(theta).unsqueeze(-1)

        sim = sim * Sigma
        attn = sim.softmax(dim=-1) * (sim > theta)
        return attn


class FA(nn.Module):
    def __init__(self, dim, window_size=(8, 8), dim_head=28, sq_dim=None, shift=True):
        super().__init__()

        if sq_dim is None:
            self.rank = dim
        else:
            self.rank = sq_dim
        self.heads_qk = sq_dim // dim_head
        self.heads_v = dim // dim_head
        self.window_size = window_size
        self.shift = shift

        num_token = window_size[0] * window_size[1]
        self.cal_atten = Attention(dim_head, num_token)

        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_qk = nn.Linear(dim, self.rank * 2, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def cal_attention(self, x):
        q, k = self.to_qk(x).chunk(2, dim=-1)
        v = self.to_v(x)
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads_qk), (q, k))
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads_v)
        attn = self.cal_atten(q, k)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

    def forward(self, x):

        b, h, w, c = x.shape
        w_size = self.window_size
        if self.shift:
            x = x.roll(shifts=4, dims=1).roll(shifts=4, dims=2)
        x_inp = rearrange(x, 'b (h b0) (w b1) c -> (b h w) (b0 b1) c', b0=w_size[0], b1=w_size[1])
        out = self.cal_attention(x_inp)
        out = rearrange(out, '(b h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size[0], w=w // w_size[1], b0=w_size[0])
        if self.shift:
            out = out.roll(shifts=-4, dims=1).roll(shifts=-4, dims=2)
        return out


class MPMLP(nn.Module):
    def __init__(self, dim, multi=4):
        super(MPMLP, self).__init__()

        self.multi = multi
        self.pwconv1 = nn.Sequential(
            nn.Conv2d(dim, dim * multi, 1, groups=dim, bias=False),
            GELU(),
        )
        self.groupconv = nn.Sequential(
            nn.Conv2d(dim * multi, dim * multi, 1, groups=multi, bias=False),
            GELU(),
        )
        self.pwconv2 = nn.Conv2d(dim * multi, dim, 1, groups=dim, bias=False)

    def forward(self, x):
        x = self.pwconv1(x.permute(0, 3, 1, 2))
        x = rearrange(x, 'b (c m) h w -> b (m c) h w', m=self.multi)
        x = self.groupconv(x)
        x = rearrange(x, 'b (m c) h w -> b (c m) h w', m=self.multi)
        x = self.pwconv2(x)
        return x.permute(0, 2, 3, 1)


class FAB(nn.Module):
    def __init__(self, dim, sq_dim, window_size=(8, 8), dim_head=28, mult=4, shift=False):
        super().__init__()

        self.pos_emb = nn.Conv2d(dim, dim, 5, 1, 2, bias=False, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.fa = FA(dim=dim, window_size=window_size, dim_head=dim_head, sq_dim=sq_dim, shift=shift)
        self.norm2 = nn.LayerNorm(dim)
        self.mpmlp = MPMLP(dim=dim, multi=mult)

    def forward(self, x):

        x = x + self.pos_emb(x)
        x = x.permute(0, 2, 3, 1)
        x_ = self.norm1(x)
        x = self.fa(x_) + x
        x_ = self.norm2(x)
        x = self.mpmlp(x_) + x
        x = x.permute(0, 3, 1, 2)
        return x


class IPB(nn.Module):
    def __init__(self, in_dim=56, out_dim=28):
        super(IPB, self).__init__()

        self.shuffle_conv = nn.Parameter(torch.cat([torch.ones(28, 1, 1, 1), torch.zeros(28, 1, 1, 1)], dim=1))
        self.conv_in = nn.Conv2d(in_dim, 28, 3, 1, 1, bias=False)
        self.down1 = FAB(dim=28, sq_dim=28, dim_head=28, mult=4)
        self.downsample1 = nn.Conv2d(28, 56, 4, 2, 1, bias=False)
        self.down2 = FAB(dim=56, sq_dim=28, dim_head=28, mult=4)
        self.downsample2 = nn.Conv2d(56, 112, 4, 2, 1, bias=False)

        self.bottleneck_local = FAB(dim=56, sq_dim=28, dim_head=28, mult=4)
        self.bottleneck_swin = FAB(dim=56, sq_dim=28, dim_head=28, mult=4, shift=True)

        self.upsample2 = nn.ConvTranspose2d(112, 56, 2, 2)
        self.fusion2 = nn.Conv2d(112, 56, 1, 1, 0, bias=False)
        self.up2 = FAB(dim=56, sq_dim=28, dim_head=28, mult=4, shift=True)
        self.upsample1 = nn.ConvTranspose2d(56, 28, 2, 2)
        self.fusion1 = nn.Conv2d(56, 28, 1, 1, 0, bias=False)
        self.up1 = FAB(dim=28, sq_dim=28, dim_head=28, mult=4, shift=True)
        self.conv_out = nn.Conv2d(28, out_dim, 3, 1, 1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        b, c, h_inp, w_inp = x.shape
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        x = rearrange(x, 'b (n c) h w -> b (c n) h w', n=2)
        x_in = F.conv2d(x, self.shuffle_conv, groups=28)

        x = self.conv_in(x)
        x1 = self.down1(x)
        x = self.downsample1(x1)
        x2 = self.down2(x)
        x = self.downsample2(x2)

        x_local = self.bottleneck_local(x[:, :56, :, :])
        x_swin = self.bottleneck_swin(x[:, 56:, :, :] + x_local)
        x = torch.cat([x_local, x_swin], dim=1)

        x = self.upsample2(x)
        x = x2 + self.fusion2(torch.cat([x, x2], dim=1))
        x = self.up2(x)
        x = self.upsample1(x)
        x = x1 + self.fusion1(torch.cat([x, x1], dim=1))
        x = self.up1(x)
        out = self.conv_out(x) + x_in

        return out[:, :, :h_inp, :w_inp]


class Mu_Estimator(nn.Module):
    def __init__(self, in_nc=28, out_nc=1, channel=32):
        super(Mu_Estimator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_nc, channel, 1, 1, 0, bias=True),
            nn.ReLU(inplace=True)
        )
        self.avpool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
            nn.Softplus())

    def forward(self, x):
        x = self.conv(x)
        x = self.avpool(x)
        x = self.mlp(x) + 1e-6
        return x


class DPB(nn.Module):
    def __init__(self, in_dim=56):
        super().__init__()
        self.norm_n = nn.LayerNorm(56)
        self.norm_mask = nn.LayerNorm(56)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_dim, 28, 1, 1, 0, bias=False),
            GELU(),
        )
        self.weight = nn.Sequential(
            nn.Conv2d(56, 28, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(28, 28, 1, 1, 0, bias=False),
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight.data, std=.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, Phi=None, Phi_compre=None):

        x = self.norm_n(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.fusion(x)
        mask = self.norm_mask(torch.cat([Phi, Phi_compre], dim=1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        weight = self.weight(mask)
        return self.out(x * weight)


class FB(nn.Module):
    def __init__(self, in_dim=28):
        super().__init__()
        self.shuffle_conv = nn.Parameter(torch.cat([torch.ones(28, 1, 1, 1), torch.zeros(28, 1, 1, 1)], dim=1))
        self.out = nn.Sequential(
            nn.Conv2d(in_dim, 28, 1, 1, 0),
            nn.GroupNorm(28, 28),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(28, 28, 1, 1, 0),
            nn.GroupNorm(28, 28),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            init.normal_(m.weight.data, mean=0.0, std=0.01)
        elif isinstance(m, nn.GroupNorm):
            init.normal_(m.weight.data, 0.0, 0.01)
            init.normal_(m.bias.data, 0.0, 0.01)

    def forward(self, f1, f2):
        f = torch.cat([f1, f2], dim=1)
        f = rearrange(f, 'b (n c) h w -> b (c n) h w', n=2)
        out = F.conv2d(f, self.shuffle_conv, groups=28) + self.out(f)
        return out


class Net(torch.nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()
        netlayer = []
        self.stage = opt.stage
        self.nC = opt.bands
        self.size = opt.size
        self.conv = nn.Conv2d(self.nC * 2, self.nC, 1, 1, 0)
        para_estimator = []
        for i in range(opt.stage):
            para_estimator.append(Mu_Estimator())

        for i in range(opt.stage):
            netlayer.append(IPB(in_dim=56))
            netlayer.append(DPB(in_dim=56))
            netlayer.append(FB(in_dim=56))

        self.mu = nn.ModuleList(para_estimator)
        self.net_stage = nn.ModuleList(netlayer)

    def reverse(self, x, len_shift=2):
        for i in range(self.nC):
            x[:, i, :, :] = torch.roll(x[:, i, :, :], shifts=(-1) * len_shift * i, dims=2)
        return x[:, :, :, :self.size]

    def shift(self, x, len_shift=2):
        x = F.pad(x, [0, self.nC*2-2, 0, 0], mode='constant', value=0)
        for i in range(self.nC):
            x[:, i, :, :] = torch.roll(x[:, i, :, :], shifts=len_shift * i, dims=2)
        return x

    def mul_PhiTg(self, Phi_shift, g):
        temp_1 = g.repeat(1, Phi_shift.shape[1], 1, 1).cuda()
        PhiTg = temp_1 * Phi_shift
        PhiTg = self.reverse(PhiTg)
        return PhiTg

    def mul_Phif(self, Phi_shift, f):
        f_shift = self.shift(f)
        Phif = Phi_shift * f_shift
        Phif = torch.sum(Phif, 1)
        return Phif.unsqueeze(1)

    def forward(self, g, input_mask=None):
        Phi, PhiPhiT = input_mask
        Phi_shift = self.shift(Phi, len_shift=2)
        Phi_compressive = torch.sum(Phi_shift, dim=1, keepdim=True)
        Phi_compressive = Phi_compressive / self.nC * 2
        Phi_compressive = self.reverse(Phi_compressive.repeat(1, 28, 1, 1))
        g_normal = g / self.nC * 2
        temp_g = g_normal.repeat(1, 28, 1, 1)
        f0 = self.reverse(temp_g)
        f = self.conv(torch.cat([f0, Phi], dim=1))
        z_ori = f
        y = 0
        r = 0

        out = []
        for i in range(self.stage):

            mu = self.mu[i](f)
            z = self.net_stage[3 * i](torch.cat([f + y / mu + r, f], dim=1))
            r = self.net_stage[3 * i + 1](torch.cat([z_ori - y / mu - f, f], dim=1), Phi, Phi_compressive)
            Phi_f = self.mul_Phif(Phi_shift, z - r - y / mu)
            f = z - r - y / mu + self.mul_PhiTg(Phi_shift, torch.div(g - Phi_f, mu + PhiPhiT))
            f = self.net_stage[3 * i + 2](f, z - r)
            z_ori = z
            y = y + mu * (f - z + r)
            out.append(f)

        return out



