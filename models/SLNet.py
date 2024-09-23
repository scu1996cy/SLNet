import math
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.distributions.normal import Normal

class SpatialTransformer_block(nn.Module):
    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode
    def forward(self, src, flow):
        shape = flow.shape[2:]
        vectors = [torch.arange(0, s) for s in shape]
        #grids = torch.meshgrid(vectors, indexing='ij')
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        grid = grid.to(flow.device)
        new_locs = grid + flow
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2,1,0]]
        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
class ResizeTransformer_block(nn.Module):
    def __init__(self, resize_factor, mode='trilinear'):
        super().__init__()
        self.factor = resize_factor
        self.mode = mode
    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x
        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
        return x

def window_partition(x, window_size):
    B, H, W, T, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], T // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0]*window_size[1]*window_size[2], C)
    return windows
def window_reverse(windows, window_size, dims):
    B, H, W, T = dims
    x = windows.view(B, H // window_size[0], W // window_size[1], T // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, H, W, T, -1)
    return x

class MHWA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.merge = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        B_, N, C = x.shape #(num_windows*B, Wh*Ww*Wt, C)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        query = query * self.scale # B_,self.num_heads, N, C // self.num_heads
        attn = (query @ key.transpose(-2, -1)) # B_,self.num_heads, N, N
        attn = self.softmax(attn)
        x = (attn @ value).transpose(1, 2).reshape(B_, N, C)
        x = self.merge(x)
        return x
class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim * 2, dim)
        self.gule1 = nn.GELU()
        self.linear2 = nn.Linear(dim, dim)
        self.dim = dim
    def forward(self, x):
        x = self.linear1(x)
        x = self.gule1(x)
        x = self.linear2(x)
        return x
class AttentionalPropagation(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=(2, 2, 2)):
        super().__init__()
        self.window_size = window_size
        self.attn = MHWA(dim, num_heads)
        self.mlp = MLP(dim)
    def forward(self, x):
        B, C, H, W, T = x.shape
        x = x.permute(0, 2, 3, 4, 1)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_f = 0
        pad_r = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_h = (self.window_size[2] - T % self.window_size[2]) % self.window_size[2]
        x = nnf.pad(x, (0, 0, pad_f, pad_h, pad_t, pad_b, pad_l, pad_r))
        _, Hp, Wp, Tp, _ = x.shape
        dims = [B, Hp, Wp, Tp]
        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, window_size, C
        message = self.attn(x_windows)
        message = message.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        message = window_reverse(message, self.window_size, dims)  # B H' W' L' C
        if pad_r > 0 or pad_b > 0:
            message = message[:, :H, :W, :T, :].contiguous()
        message = message.view(B, H * W * T, C)
        x = x.view(B, H * W * T, C)
        x_out = self.mlp(torch.cat([x, message], dim=2))
        xout = x_out.view(-1, H, W, T, C).permute(0, 4, 1, 2, 3).contiguous()
        return xout

class DualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.act1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.act2 = nn.LeakyReLU(0.1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x_out = self.act2(x)
        return x_out
class ConvAttEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.act1 = nn.LeakyReLU(0.1)
        self.trans = AttentionalPropagation(out_channels)
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x_out = self.trans(x)
        return x_out
class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.deconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.act = nn.LeakyReLU(0.1)
    def forward(self, x):
        x = self.deconv(x)
        x = self.norm(x)
        x_out = self.act(x)
        return x_out
class SharedEncoder(nn.Module):
    def __init__(self, in_channels=1, channel_num=16):
        super().__init__()
        self.conv_1 = DualConvBlock(in_channels, channel_num)
        self.conv_2 = ConvAttEncoderBlock(channel_num, channel_num * 2)
        self.conv_3 = ConvAttEncoderBlock(channel_num * 2, channel_num * 4)
        self.conv_4 = ConvAttEncoderBlock(channel_num * 4, channel_num * 8)
        self.conv_5 = ConvAttEncoderBlock(channel_num * 8, channel_num * 16)
        self.downsample = nn.AvgPool3d(2, stride=2)
    def forward(self, x_in):
        x_1 = self.conv_1(x_in)
        x = self.downsample(x_1)
        x_2 = self.conv_2(x)
        x = self.downsample(x_2)
        x_3 = self.conv_3(x)
        x = self.downsample(x_3)
        x_4 = self.conv_4(x)
        x = self.downsample(x_4)
        x_5 = self.conv_5(x)
        return [x_1, x_2, x_3, x_4, x_5]

def get_winsize(x_size, window_size):
    use_window_size = list(window_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
    return tuple(use_window_size)
def get_aff(xk, yk):
    b, n, c = xk.shape
    xk = xk.permute(0, 2, 1) #b, c, n
    yk = yk.permute(0, 2, 1) #b, c, n
    a_sq = xk.pow(2).sum(1).unsqueeze(2)
    ab = xk.transpose(1, 2) @ yk
    affinity = (2 * ab - a_sq) / math.sqrt(c)
    # softmax operation; aligned the evaluation style
    maxes = torch.max(affinity, dim=1, keepdim=True)[0]
    x_exp = torch.exp(affinity - maxes)
    x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
    affinity = x_exp / x_exp_sum
    affinity = affinity.permute(0, 2, 1)
    return affinity
class Sim(nn.Module):
    def __init__(self, dim, window_size=[2, 2, 2]):
        super().__init__()
        self.channel = window_size[0] * window_size[1] * window_size[2]
        self.window_size = window_size
        self.normx = nn.LayerNorm(dim)
        self.normy = nn.LayerNorm(dim)
        vectors = [torch.arange(-s // 2 + 1, s // 2 + 1) for s in window_size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids, -1).type(torch.FloatTensor)
        self.register_buffer('grid', grid)
    def makeV(self, N):
        v = self.grid.reshape(self.channel, 3).unsqueeze(0).repeat(N, 1, 1).unsqueeze(0)
        return v
    def forward(self, x_in, y_in):
        b, c, d, h, w = x_in.shape
        n = d*h*w
        x = x_in.permute(0, 2, 3, 4, 1)
        y = y_in.permute(0, 2, 3, 4, 1)
        x = self.normx(x)
        y = self.normy(y)
        window_size = get_winsize((d, h, w), self.window_size)
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
        x = nnf.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        y = nnf.pad(y, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, dp, hp, wp, _ = x.shape
        dims = [b, dp, hp, wp]
        x_windows = window_partition(x, window_size)
        y_windows = window_partition(y, window_size)
        affinity = get_aff(x_windows, y_windows) # b_, n_, c_
        affinity = affinity.view(-1, *(window_size + (self.channel,)))
        affinity = window_reverse(affinity, window_size, dims)
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            affinity = affinity[:, :d, :h, :w, :].contiguous()
        affinity = affinity.view(b, n, self.channel).reshape(b, n, self.channel, 1).transpose(2, 3) # b, n, 1, 8
        v = self.makeV(n)  # b, n, 8, 3
        out = (affinity @ v)  # b, n, 1, 3
        out = out.reshape(b, d, h, w, 3).permute(0, 4, 1, 2, 3)
        #out = out.reshape(b, d, h, w, 3).permute(0, 4, 1, 2, 3).contiguous()
        return out

class RegHead(nn.Module):
    def __init__(self, in_channels, out_channels=3, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.reg_head = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.reg_head.weight = nn.Parameter(Normal(0, 1e-5).sample(self.reg_head.weight.shape))
        self.reg_head.bias = nn.Parameter(torch.zeros(self.reg_head.bias.shape))
    def forward(self, x):
        x_out = self.reg_head(x)
        return x_out
class VecInt(nn.Module):
    def __init__(self, nsteps=7):
        super().__init__()
        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer_block()
    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

class SLNet(nn.Module):
    def __init__(self, in_channel=16, channel_num=16):
        super().__init__()

        in_channel = in_channel
        self.encoder = SharedEncoder(channel_num=in_channel)

        self.conv_1 = DualConvBlock(in_channel * 1 * 2 + channel_num * 1 + 3, channel_num * 1)
        self.conv_2 = DualConvBlock(in_channel * 2 * 2 + channel_num * 2 + 3, channel_num * 2)
        self.conv_3 = DualConvBlock(in_channel * 4 * 2 + channel_num * 4 + 3, channel_num * 4)
        self.conv_4 = DualConvBlock(in_channel * 8 * 2 + channel_num * 8 + 3, channel_num * 8)
        self.conv_5 = DualConvBlock(in_channel * 16 * 2 + 3, channel_num * 16)

        self.corr_1 = Sim(in_channel * 1)
        self.corr_2 = Sim(in_channel * 2)
        self.corr_3 = Sim(in_channel * 4)
        self.corr_4 = Sim(in_channel * 8)
        self.corr_5 = Sim(in_channel * 16)

        self.upsample_1 = DeconvBlock(channel_num * 2, channel_num * 1)
        self.upsample_2 = DeconvBlock(channel_num * 4, channel_num * 2)
        self.upsample_3 = DeconvBlock(channel_num * 8, channel_num * 4)
        self.upsample_4 = DeconvBlock(channel_num * 16, channel_num * 8)

        self.reghead_1 = RegHead(channel_num * 1)
        self.reghead_2 = RegHead(channel_num * 2)
        self.reghead_3 = RegHead(channel_num * 4)
        self.reghead_4 = RegHead(channel_num * 8)
        self.reghead_5 = RegHead(channel_num * 16)

        self.resize_transformer = nn.ModuleList()
        self.spatial_transformer = nn.ModuleList()
        for i in range(5):
            self.resize_transformer.append(ResizeTransformer_block(resize_factor=2, mode='trilinear'))
            self.spatial_transformer.append(SpatialTransformer_block(mode='bilinear'))

    def forward(self, moving, fixed):
        x_mov_1, x_mov_2, x_mov_3, x_mov_4, x_mov_5 = self.encoder(moving)
        x_fix_1, x_fix_2, x_fix_3, x_fix_4, x_fix_5 = self.encoder(fixed)

        corr_5 = self.corr_5(x_mov_5, x_fix_5)
        cat = torch.cat([x_mov_5, corr_5, x_fix_5], dim=1)
        conv_corr_5 = self.conv_5(cat)
        flow_5 = self.reghead_5(conv_corr_5)

        # Step 4
        flow_5_up = self.resize_transformer[3](flow_5)
        x_mov_4 = self.spatial_transformer[3](x_mov_4, flow_5_up)

        conv_corr_5_up = self.upsample_4(conv_corr_5)
        corr_4 = self.corr_4(x_mov_4, x_fix_4)
        cat = torch.cat([x_mov_4, corr_4, x_fix_4, conv_corr_5_up], dim=1)
        conv_corr_4 = self.conv_4(cat)
        flow_4 = self.reghead_4(conv_corr_4)
        flow_4 = flow_4 + flow_5_up

        # Step 3
        flow_4_up = self.resize_transformer[2](flow_4)
        x_mov_3 = self.spatial_transformer[2](x_mov_3, flow_4_up)

        conv_corr_4_up = self.upsample_3(conv_corr_4)
        corr_3 = self.corr_3(x_mov_3, x_fix_3)
        cat = torch.cat([x_mov_3, corr_3, x_fix_3, conv_corr_4_up], dim=1)
        conv_corr_3 = self.conv_3(cat)
        flow_3 = self.reghead_3(conv_corr_3)
        flow_3 = flow_3 + flow_4_up

        # Step 2
        flow_3_up = self.resize_transformer[1](flow_3)
        x_mov_2 = self.spatial_transformer[1](x_mov_2, flow_3_up)

        conv_corr_3_up = self.upsample_2(conv_corr_3)
        corr_2 = self.corr_2(x_mov_2, x_fix_2)
        cat = torch.cat([x_mov_2, corr_2, x_fix_2, conv_corr_3_up], dim=1)
        conv_corr_2 = self.conv_2(cat)
        flow_2 = self.reghead_2(conv_corr_2)
        flow_2 = flow_2 + flow_3_up

        # Step 1
        flow_2_up = self.resize_transformer[0](flow_2)
        x_mov_1 = self.spatial_transformer[0](x_mov_1, flow_2_up)

        conv_corr_2_up = self.upsample_1(conv_corr_2)
        corr_1 = self.corr_1(x_mov_1, x_fix_1)
        cat = torch.cat([x_mov_1, corr_1, x_fix_1, conv_corr_2_up], dim=1)
        conv_corr_1 = self.conv_1(cat)
        flow_1 = self.reghead_1(conv_corr_1)
        flow_1 = flow_1 + flow_2_up

        moved = self.spatial_transformer[0](moving, flow_1)
        return moved, flow_1
class SLNet_diff(nn.Module):
    def __init__(self, in_channel=16):
        super().__init__()

        in_channel = in_channel
        self.encoder = SharedEncoder(channel_num=in_channel)

        self.corr_1 = Sim(in_channel * 1)
        self.corr_2 = Sim(in_channel * 2)
        self.corr_3 = Sim(in_channel * 4)
        self.corr_4 = Sim(in_channel * 8)
        self.corr_5 = Sim(in_channel * 16)

        self.reghead_1 = RegHead(3)
        self.reghead_2 = RegHead(3)
        self.reghead_3 = RegHead(3)
        self.reghead_4 = RegHead(3)
        self.reghead_5 = RegHead(3)

        self.resize_transformer = nn.ModuleList()
        self.spatial_transformer = nn.ModuleList()
        self.integrate = nn.ModuleList()
        for i in range(5):
            self.resize_transformer.append(ResizeTransformer_block(resize_factor=2, mode='trilinear'))
            self.spatial_transformer.append(SpatialTransformer_block(mode='bilinear'))
            self.integrate.append(VecInt())
    def forward(self, moving, fixed):
        x_mov_1, x_mov_2, x_mov_3, x_mov_4, x_mov_5 = self.encoder(moving)
        x_fix_1, x_fix_2, x_fix_3, x_fix_4, x_fix_5 = self.encoder(fixed)

        corr_5 = self.corr_5(x_mov_5, x_fix_5)
        flow_5 = self.reghead_5(corr_5)
        flow_5 = self.integrate[4](flow_5)

        # Step 4
        flow_5_up = self.resize_transformer[3](flow_5)
        x_mov_4 = self.spatial_transformer[3](x_mov_4, flow_5_up)

        corr_4 = self.corr_4(x_mov_4, x_fix_4)
        flow_4 = self.reghead_4(corr_4)
        flow_4 = self.integrate[3](flow_4)
        flow_4 = flow_4 + flow_5_up

        # Step 3
        flow_4_up = self.resize_transformer[2](flow_4)
        x_mov_3 = self.spatial_transformer[2](x_mov_3, flow_4_up)

        corr_3 = self.corr_3(x_mov_3, x_fix_3)
        flow_3 = self.reghead_3(corr_3)
        flow_3 = self.integrate[2](flow_3)
        flow_3 = flow_3 + flow_4_up

        # Step 2
        flow_3_up = self.resize_transformer[1](flow_3)
        x_mov_2 = self.spatial_transformer[1](x_mov_2, flow_3_up)

        corr_2 = self.corr_2(x_mov_2, x_fix_2)
        flow_2 = self.reghead_2(corr_2)
        flow_2 = self.integrate[1](flow_2)
        flow_2 = flow_2 + flow_3_up

        # Step 1
        flow_2_up = self.resize_transformer[0](flow_2)
        x_mov_1 = self.spatial_transformer[0](x_mov_1, flow_2_up)

        corr_1 = self.corr_1(x_mov_1, x_fix_1)
        flow_1 = self.reghead_1(corr_1)
        flow_1 = self.integrate[0](flow_1)
        flow_1 = flow_1 + flow_2_up

        moved = self.spatial_transformer[0](moving, flow_1)
        return moved, flow_1

class SLNet_flow(nn.Module):
    def __init__(self, in_channel=16, channel_num=16):
        super().__init__()

        in_channel = in_channel
        self.encoder = SharedEncoder(channel_num=in_channel)

        self.conv_1 = DualConvBlock(in_channel * 1 * 2 + channel_num * 1, channel_num * 1)
        self.conv_2 = DualConvBlock(in_channel * 2 * 2 + channel_num * 2 + 3, channel_num * 2)
        self.conv_3 = DualConvBlock(in_channel * 4 * 2 + channel_num * 4 + 3, channel_num * 4)
        self.conv_4 = DualConvBlock(in_channel * 8 * 2 + channel_num * 8 + 3, channel_num * 8)
        self.conv_5 = DualConvBlock(in_channel * 16 * 2 + 3, channel_num * 16)

        self.corr_2 = Sim(in_channel * 2)
        self.corr_3 = Sim(in_channel * 4)
        self.corr_4 = Sim(in_channel * 8)
        self.corr_5 = Sim(in_channel * 16)

        self.upsample_1 = DeconvBlock(channel_num * 2, channel_num * 1)
        self.upsample_2 = DeconvBlock(channel_num * 4, channel_num * 2)
        self.upsample_3 = DeconvBlock(channel_num * 8, channel_num * 4)
        self.upsample_4 = DeconvBlock(channel_num * 16, channel_num * 8)

        self.reghead_1 = RegHead(channel_num * 1)
        self.reghead_2 = RegHead(channel_num * 2)
        self.reghead_3 = RegHead(channel_num * 4)
        self.reghead_4 = RegHead(channel_num * 8)
        self.reghead_5 = RegHead(channel_num * 16)

        self.resize_transformer = nn.ModuleList()
        self.spatial_transformer = nn.ModuleList()
        for i in range(4):
            self.resize_transformer.append(ResizeTransformer_block(resize_factor=2, mode='trilinear'))
            self.spatial_transformer.append(SpatialTransformer_block(mode='bilinear'))

    def forward(self, moving, fixed):
        x_mov_1, x_mov_2, x_mov_3, x_mov_4, x_mov_5 = self.encoder(moving)
        x_fix_1, x_fix_2, x_fix_3, x_fix_4, x_fix_5 = self.encoder(fixed)

        corr_5 = self.corr_5(x_mov_5, x_fix_5)
        cat = torch.cat([x_mov_5, corr_5, x_fix_5], dim=1)
        conv_corr_5 = self.conv_5(cat)
        flow_5 = self.reghead_5(conv_corr_5)

        # Step 4
        flow_5_up = self.resize_transformer[3](flow_5)
        x_mov_4 = self.spatial_transformer[3](x_mov_4, flow_5_up)

        conv_corr_5_up = self.upsample_4(conv_corr_5)
        corr_4 = self.corr_4(x_mov_4, x_fix_4)
        cat = torch.cat([x_mov_4, corr_4, x_fix_4, conv_corr_5_up], dim=1)
        conv_corr_4 = self.conv_4(cat)
        flow_4 = self.reghead_4(conv_corr_4)
        flow_4 = self.spatial_transformer[3](flow_5_up, flow_4)+flow_4

        # Step 3
        flow_4_up = self.resize_transformer[2](flow_4)
        x_mov_3 = self.spatial_transformer[2](x_mov_3, flow_4_up)

        conv_corr_4_up = self.upsample_3(conv_corr_4)
        corr_3 = self.corr_3(x_mov_3, x_fix_3)
        cat = torch.cat([x_mov_3, corr_3, x_fix_3, conv_corr_4_up], dim=1)
        conv_corr_3 = self.conv_3(cat)
        flow_3 = self.reghead_3(conv_corr_3)
        flow_3 = self.spatial_transformer[2](flow_4_up, flow_3)+flow_3

        # Step 2
        flow_3_up = self.resize_transformer[1](flow_3)
        x_mov_2 = self.spatial_transformer[1](x_mov_2, flow_3_up)

        conv_corr_3_up = self.upsample_2(conv_corr_3)
        corr_2 = self.corr_2(x_mov_2, x_fix_2)
        cat = torch.cat([x_mov_2, corr_2, x_fix_2, conv_corr_3_up], dim=1)
        conv_corr_2 = self.conv_2(cat)
        flow_2 = self.reghead_2(conv_corr_2)
        flow_2 = self.spatial_transformer[1](flow_3_up, flow_2)+flow_2

        # Step 1
        flow_2_up = self.resize_transformer[0](flow_2)
        x_mov_1 = self.spatial_transformer[0](x_mov_1, flow_2_up)

        conv_corr_2_up = self.upsample_1(conv_corr_2)
        cat = torch.cat([x_mov_1, x_fix_1, conv_corr_2_up], dim=1)
        conv_corr_1 = self.conv_1(cat)
        flow_1 = self.reghead_1(conv_corr_1)
        flow_1 = self.spatial_transformer[0](flow_2_up, flow_1)+flow_1

        moved = self.spatial_transformer[0](moving, flow_1)
        return moved, flow_1
class SLNet_flow_diff(nn.Module):
    def __init__(self, in_channel=16):
        super().__init__()

        in_channel = in_channel
        self.encoder = SharedEncoder(channel_num=in_channel)

        self.corr_1 = Sim(in_channel * 1)
        self.corr_2 = Sim(in_channel * 2)
        self.corr_3 = Sim(in_channel * 4)
        self.corr_4 = Sim(in_channel * 8)
        self.corr_5 = Sim(in_channel * 16)

        self.reghead_1 = RegHead(3)
        self.reghead_2 = RegHead(3)
        self.reghead_3 = RegHead(3)
        self.reghead_4 = RegHead(3)
        self.reghead_5 = RegHead(3)

        self.resize_transformer = nn.ModuleList()
        self.spatial_transformer = nn.ModuleList()
        self.integrate = nn.ModuleList()
        for i in range(5):
            self.resize_transformer.append(ResizeTransformer_block(resize_factor=2, mode='trilinear'))
            self.spatial_transformer.append(SpatialTransformer_block(mode='bilinear'))
            self.integrate.append(VecInt())
    def forward(self, moving, fixed):
        x_mov_1, x_mov_2, x_mov_3, x_mov_4, x_mov_5 = self.encoder(moving)
        x_fix_1, x_fix_2, x_fix_3, x_fix_4, x_fix_5 = self.encoder(fixed)

        corr_5 = self.corr_5(x_mov_5, x_fix_5)
        flow_5 = self.reghead_5(corr_5)
        flow_5 = self.integrate[4](flow_5)

        # Step 4
        flow_5_up = self.resize_transformer[3](flow_5)
        x_mov_4 = self.spatial_transformer[3](x_mov_4, flow_5_up)

        corr_4 = self.corr_4(x_mov_4, x_fix_4)
        flow_4 = self.reghead_4(corr_4)
        flow_4 = self.integrate[3](flow_4)
        flow_4 = self.spatial_transformer[3](flow_5_up, flow_4)+flow_4

        # Step 3
        flow_4_up = self.resize_transformer[2](flow_4)
        x_mov_3 = self.spatial_transformer[2](x_mov_3, flow_4_up)

        corr_3 = self.corr_3(x_mov_3, x_fix_3)
        flow_3 = self.reghead_3(corr_3)
        flow_3 = self.integrate[2](flow_3)
        flow_3 = self.spatial_transformer[2](flow_4_up, flow_3)+flow_3

        # Step 2
        flow_3_up = self.resize_transformer[1](flow_3)
        x_mov_2 = self.spatial_transformer[1](x_mov_2, flow_3_up)

        corr_2 = self.corr_2(x_mov_2, x_fix_2)
        flow_2 = self.reghead_2(corr_2)
        flow_2 = self.integrate[1](flow_2)
        flow_2 = self.spatial_transformer[1](flow_3_up, flow_2)+flow_2

        # Step 1
        flow_2_up = self.resize_transformer[0](flow_2)
        x_mov_1 = self.spatial_transformer[0](x_mov_1, flow_2_up)

        corr_1 = self.corr_1(x_mov_1, x_fix_1)
        flow_1 = self.reghead_1(corr_1)
        flow_1 = self.integrate[0](flow_1)
        flow_1 = self.spatial_transformer[0](flow_2_up, flow_1)+flow_1

        moved = self.spatial_transformer[0](moving, flow_1)
        return moved, flow_1
