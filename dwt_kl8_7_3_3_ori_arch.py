import torch
import torch.nn as nn
import torch.nn.functional as F
import sys 
sys.path.append("/code/UHDformer-main")
from basicsr.archs.VAE_arch import AutoencoderKL
import time
import yaml
from basicsr.utils.vae_util import instantiate_from_config
# from utils.distributions.distributions import DiagonalGaussianDistribution
from basicsr.utils.registry import ARCH_REGISTRY
import math
from basicsr.utils.distributions.distributions import DiagonalGaussianDistribution
from basicsr.archs.encoder import nonlinearity, Normalize, ResnetBlock, make_attn, Downsample, Upsample
from basicsr.archs.wtconv import WTConv2d
from einops import rearrange
from basicsr.archs.Fourier_Upsampling import freup_Areadinterpolation,freup_AreadinterpolationV2,freup_Cornerdinterpolation,freup_Periodicpadding
from basicsr.archs.wtconv.util import wavelet
from basicsr.archs.merge.gate import GatedFeatureEnhancement
from basicsr.archs.Resblock.Res_four import Res_four,Res_four2,Res_four3,Res_four4,Res_four5,Res_four6,Res_four7,Res_four8,Res_four9,Res_four10,Res_four11,Res_four12

import numbers
import numpy as np  

import torch.fft as fft

# Layer Norm
class fresadd(nn.Module):
    def __init__(self, channels=32,freup_type='pad'):
        super(fresadd, self).__init__()
        if freup_type == 'pad':
            self.Fup = freup_Periodicpadding(channels)
        elif freup_type == 'corner':
            self.Fup = freup_Cornerdinterpolation(channels)
        elif freup_type == 'area':
            self.Fup = freup_Areadinterpolation(channels)
        elif freup_type == 'areaV2':
            self.Fup = freup_AreadinterpolationV2(channels)
        print('freup_type is',freup_type)
        self.fuse = nn.Conv2d(channels, channels,1,1,0)

    def forward(self,x):

        x1 = x
        
        x2 = F.interpolate(x1,scale_factor=2,mode='bilinear')
      
        x3 = self.Fup(x1)
     


        xm = x2 + x3
        xn = self.fuse(xm)

        return xn

def make_res(in_channels, out_channels,temb_channels,dropout, res_type="vanilla"):
    assert res_type in ["vanilla", "Fourmer","MAB","Res_four","Res_four2","Res_four3","Res_four4","Res_four5","Res_four6","Res_four7","Res_four8","Res_four9","Res_four10","Res_four11","Res_four12","none"], f'res_type {res_type} unknown'
    print(f"making res of type '{res_type}' with {in_channels} in_channels")
    if res_type == "vanilla":
        return ResnetBlock(in_channels=in_channels,
                                         out_channels=out_channels,
                                         temb_channels=temb_channels,
                                         dropout=dropout)
    elif res_type == "Fourmer":
        return ProcessBlock(in_channels,out_channels)
    elif res_type == "Res_four":
        return Res_four(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)
    elif res_type == "Res_four2":
        return Res_four2(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)
    elif res_type == "Res_four3":
        return Res_four3(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)
    elif res_type == "Res_four4":
        return Res_four4(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)
    elif res_type == "Res_four5":
        return Res_four5(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)
    elif res_type == "Res_four6":
        return Res_four6(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)
    elif res_type == "Res_four7":
        return Res_four7(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)
    elif res_type == "Res_four8":
        return Res_four8(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)
    elif res_type == "Res_four9":
        return Res_four9(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)
    elif res_type == "Res_four10":
        return Res_four10(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)
    elif res_type == "Res_four11":
        return Res_four11(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)
    elif res_type == "Res_four12":
        return Res_four12(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)
    elif res_type == "MAB":
        return MAB(in_channels, out_channels)

    else:
        return nn.Identity(in_channels)

# Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
    

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
    



class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# FC
class FC(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.fc = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(), 
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.fc(x)


# Local feature
class Local(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        hidden_dim = int(dim // growth_rate)

        self.weight = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.weight(y)
        return x*y


# Gobal feature
class Gobal(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act3 = nn.Sigmoid()

    def forward(self, x):
        _, C, H, W = x.shape
        y = F.interpolate(x, size=[C, C], mode='bilinear', align_corners=True) #[1, 64, 64, 64]
        # b c w h -> b c h w
        y = self.act1(self.conv1(y)).permute(0, 1, 3, 2)
        # b c h w -> b w h c
        y = self.act2(self.conv2(y)).permute(0, 3, 2, 1)
        # b w h c -> b c w h
        y = self.act3(self.conv3(y)).permute(0, 3, 1, 2)
        y = F.interpolate(y, size=[H, W], mode='bilinear', align_corners=True)
        return x*y
    

class AttBlock(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = LayerNorm(dim) 
        self.norm2 = LayerNorm(dim) 

        self.local = Local(dim, ffn_scale)
        self.gobal = Gobal(dim)
        self.conv = nn.Conv2d(2*dim, dim, 1, 1, 0)
        # Feedforward layer
        self.fc = FC(dim, ffn_scale) 

    def forward(self, x):
        y = self.norm1(x)
        y_l = self.local(y)
        y_g = self.gobal(y)
        y = self.conv(torch.cat([y_l, y_g], dim=1)) + x

        y = self.fc(self.norm2(y)) + y
        return y
    

class ResBlock(nn.Module):
    def __init__(self, dim, k=3, s=1, p=1, b=True):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=k, stride=s, padding=p, bias=b)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=k, stride=s, padding=p, bias=b)

    def forward(self, x):
        res = self.conv2(self.act(self.conv1(x)))
        return res + x

@ARCH_REGISTRY.register()
class dwt_kl8_7_3_3_ori(nn.Module):
    def __init__(self, dim, n_blocks=8, ffn_scale=2.0, upscaling_factor=4,vae_weight=None,config=None,sample= True,dwt_dim = 16,num_heads=3,param_key = 'params',out_dim= 64):
        super().__init__()
        with open(config) as f:
            # config = yaml.load(f, Loader=yaml.FullLoader)
            config = yaml.load(f, Loader=yaml.FullLoader)["network_g"]
            config.pop('type')
            self.vae = AutoencoderKL(**config,dwt_dim=dwt_dim,num_heads=num_heads)
            self.sample = sample
            if vae_weight:
                print(torch.load(vae_weight,map_location='cpu').keys())
                msg = self.vae.load_state_dict(torch.load(vae_weight,map_location='cpu')[param_key],strict=False)
                print(f"load vae weight from{param_key}")
                print(f"load vae weight from {vae_weight}")
                print('missing keys:',len(msg.missing_keys),'unexpected keys:',len(msg.unexpected_keys))
                
           
            a =0
            for name, param in self.vae.named_parameters():
                if name in msg.missing_keys:
                    param.requires_grad = True
                    a +=1
                    

                else:
                    param.requires_grad = False
            print(f"adapter num is {a}")
            assert len(msg.missing_keys) == a


        self.feats = nn.Sequential(*[AttBlock(dim, ffn_scale) for _ in range(n_blocks)])

        self.rec_block = nn.Sequential(WTConv2d(3, 3, kernel_size=5, stride=1, bias=True, wt_levels=3, wt_type='db1'),
                                       nn.Conv2d(3, 16, 1, 1, 0),
                                       nn.Conv2d(16, 3, 1, 1, 0),
                                       WTConv2d(3, 3, kernel_size=5, stride=1, bias=True, wt_levels=3, wt_type='db1')
                                       )
        

    def forward(self, input):
        x,add,high_list = self.vae.encode(input)
        posterior = DiagonalGaussianDistribution(x)
        if self.sample:
            x = posterior.sample()
        else:
            x = posterior.mode()

        
        x = self.feats(x) + x #[1, 64, 240, 135]
        
        input = input + self.rec_block(input)
        
        x = self.vae.decode(x,add,high_list) + input

        

        return x
        # return x, posterior
    

    @torch.no_grad()
    def test_tile(self, input, tile_size=512, tile_pad=16):
        # return self.test(input)
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
        """
        batch, channel, height, width = input.shape
        output_height = height 
        output_width = width 
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = input.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = input[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                output_tile = self(input_tile)

                # output tile area on total image
                output_start_x = input_start_x 
                output_end_x = input_end_x 
                output_start_y = input_start_y 
                output_end_y = input_end_y 

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) 
                output_end_x_tile = output_start_x_tile + input_tile_width 
                output_start_y_tile = (input_start_y - input_start_y_pad) 
                output_end_y_tile = output_start_y_tile + input_tile_height 

                # put tile into output image
                output[:, :, output_start_y:output_end_y,
                output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                               output_start_x_tile:output_end_x_tile]
        return output
    



class AutoencoderKL(nn.Module):
    def __init__(self,
                 ddconfig,    
                 embed_dim,
                 optim,
                 dwt_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 num_heads=3,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig,dwt_dim=dwt_dim)
        self.decoder = Decoder(**ddconfig,dwt_dim=dwt_dim)
        # self.loss = instantiate_from_config(lossconfig)
        self.adapter_mid = nn.Sequential(make_attn(dwt_dim, attn_type="restormer",num_heads=num_heads),
                                         nn.Conv2d(dwt_dim, dwt_dim, 1),
                                     make_attn(dwt_dim, attn_type="restormer",num_heads=num_heads))
        self.learning_rate = optim["lr"]
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h,ll ,high_list = self.encoder(x)
        ll = self.adapter_mid(ll)
        moments = self.quant_conv(h)
        
        return moments,ll,high_list

    def decode(self, z,ll,high_list):
        z = self.post_quant_conv(z)
        dec = self.decoder(z,ll,high_list)
        return dec

    def forward(self, input):
        #采样过程合并到encode中
        z,posterior = self.encode(input)
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        # x = batch[k]
        x = batch
        # print(x.shape)
        if len(x.shape) == 3:
            x = x[..., None]
        # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x
        
        

        
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, freup_type='pad', in_channels,
                 resolution, z_channels, dwt_dim = 16,give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="restormer",res_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = make_res(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,res_type=res_type)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = make_res(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,res_type=res_type)

        # upsampling
        self.up = nn.ModuleList()
        self.adapters = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(make_res(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,res_type=res_type))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = fresadd(block_in, freup_type)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order
            self.adapters.insert(0, dwt_revadapter(dwt_dim,dwt_dim, block_out))

        # end
        self.norm_out = Normalize(block_in)
        if dwt_dim > 4:
            
            self.norm_out2 = Normalize(dwt_dim)
        else:
            self.norm_out2 = Normalize(dwt_dim,num_groups=dwt_dim)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        
        self.conv_out2 = torch.nn.Conv2d(dwt_dim,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z,ll,high_list):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)
        add_in = ll
        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
                add,add_in = self.adapters[i_level](add_in,high_list[i_level-1],h)
                h = h + add

                

        # end
        if self.give_pre_end:
            return h
        det_out = self.conv_out2(nonlinearity(self.norm_out2(add_in)))
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
            det_out = torch.tanh(det_out)
        return h+det_out

class ZeroConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ZeroConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)

class Adapter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Adapter, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels//4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels//4, out_channels)
        self.out_channels = out_channels
        self.init_weights()

    def init_weights(self):
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x_flat = x.view(batch_size, channels, -1).permute(0, 2, 1)  # Reshape to (batch_size, height*width, channels)
        x_flat = self.fc1(x_flat)
        x_flat = self.relu(x_flat)
        x_flat = self.fc2(x_flat)
        x_flat = x_flat.permute(0, 2, 1).view(batch_size, self.out_channels, height, width)  # Reshape back
        return  x_flat
    
class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None
    
    def forward(self, x):
        return torch.mul(self.weight, x)

class FFT_filter(nn.Module):
    def __init__(self, C1, C2):
        super(FFT_filter, self).__init__()
        
        self.C1 = C1
        self.C2 = C2
        

        self.filter_generators = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(C1, C2, kernel_size=1),
                nn.Sigmoid()  
            ) for _ in range(4)
        ])
        

        self.channel_weight_generator = nn.Sequential(
            nn.Conv2d(C2, 4*C1, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )
        self.fuse = nn.Conv2d(4*C2, C2, 1)

        self.output1_conv = nn.Conv2d(4 * C1 + C2, C2, kernel_size=1)
        self.output2_conv = nn.Conv2d(4 * C1 + C2, 1 * C1, kernel_size=1)
        self.output3_conv = nn.Conv2d(4 * C1 + C2, 3 * C1, kernel_size=1)

    def forward(self, x1, x2):
        # x1: (B, 4*C1, H, W)
        # x2: (B, C2, H, W)
        B, _, H, W = x1.shape
        
        x1_splits = torch.split(x1, self.C1, dim=1) 
        
        x2_rfft = fft.rfft2(x2)
        x2_rfft_shifted = fft.fftshift(x2_rfft)  
        
        outputs2 = []
        for i in range(4):
            filters = self.filter_generators[i](x1_splits[i])
            
            half_width = W // 2 +1
            filters_first_half = filters[..., :half_width]
            filters_second_half = torch.flip(filters[..., half_width-2:],dims=[-1])


            filters_avg = (filters_first_half + filters_second_half) / 2
            
            filtered_rfft = x2_rfft_shifted * filters_avg
            output_irfft = fft.irfft2(fft.ifftshift(filtered_rfft), s=(H, W))
            output_irfft = output_irfft + x2
            outputs2.append(output_irfft)

        output2 = torch.cat(outputs2, dim=1)
        output2 = self.fuse(output2)
        

        channel_weight = self.channel_weight_generator(x2)
        output1 = x1 * channel_weight+x1


        fused_feature = torch.cat([output1, output2], dim=1)

        return fused_feature

class WTblock(nn.Module):
    def __init__(self, in_channels, out_channels,enc_channel, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTblock, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter = wavelet.create_wavelet_decfilter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        
        self.wt_function = wavelet.wavelet_transform_init(self.wt_filter)
        
        # self.merge = nn.Conv2d(in_channels*4+enc_channel, in_channels*4, kernel_size=3, padding=1)
        # self.merge = GatedFeatureEnhancement(in_channels*4,enc_channel)
        self.naf = NAFBlock(in_channels*4)
        self.wavelet_convs = nn.Conv2d(in_channels*4, in_channels*4, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels*4, bias=False) 
        self.wavelet_scale = _ScaleModule([1,in_channels*4,1,1], init_scale=0.1) 
        self.merge = FFT_filter(in_channels,enc_channel)
        
        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride, groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x,enc): 
        curr_x_ll = x
        curr_shape = curr_x_ll.shape
        if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)
    
        curr_x = self.wt_function(curr_x_ll)
            
        shape_x = curr_x.shape
        curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
        # curr_x_tag = torch.cat((curr_x_tag,enc),dim=1)
        # curr_x_tag = self.merge(curr_x_tag,enc)
        curr_x_tag = self.naf(curr_x_tag)
        curr_x_tag = self.wavelet_convs(curr_x_tag)

        curr_x_tag = self.wavelet_scale(curr_x_tag)
        curr_x_tag = self.merge(curr_x_tag,enc)

        

        return curr_x_tag
    
class WT_revblock(nn.Module):
    def __init__(self, in_channels, out_channels,enc_channel, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WT_revblock, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.iwt_filter = wavelet.create_wavelet_recfilter(wt_type, in_channels, in_channels, torch.float)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        
        self.iwt_function = wavelet.inverse_wavelet_transform_init(self.iwt_filter)
        self.merge_conv  = nn.Conv2d(in_channels+enc_channel, in_channels, kernel_size=3, padding=1)
        self.wavelet_scale = _ScaleModule([1,in_channels*4,1,1], init_scale=0.1) 
        self.naf = NAFBlock(in_channels)
        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride, groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, ll,high ,enc,ll_before= None): 
        if ll_before:
            ll = ll+ll_before
        curr_x = torch.cat([ll.unsqueeze(2), high], dim=2)
        next_x_ll = self.iwt_function(curr_x)
        next_x_ll = torch.cat([next_x_ll,enc],dim=1)
        next_x_ll = self.merge_conv(next_x_ll)
        next_x_ll = self.naf(next_x_ll)
        

        return next_x_ll
    

class dwt_adapter (nn.Module):
    def __init__(self, in_channels, out_channels,enc_channel, kernel_size=5, stride=1):
        super(dwt_adapter, self).__init__()

        self.wtblock = WTblock(in_channels, out_channels,enc_channel, kernel_size=5, stride=1, bias=True, wt_levels=2, wt_type='db1')
        self.zeroconv = ZeroConv2d(enc_channel, enc_channel, 1)
        self.enc_conv = nn.Conv2d(in_channels*4+enc_channel, enc_channel, 3,padding=1, stride=stride)
        self.ll_conv = nn.Conv2d(in_channels*4+enc_channel,in_channels,1)
        self.high_conv = nn.Conv2d(in_channels*4+enc_channel,in_channels*3,1)

    def forward(self, x,enc):
        x = self.wtblock(x,enc)
        b,c,h,w = x.shape
        n = 4
        c = int((c -enc.shape[1])/n)
        
        # x = x.view(b,-1,h,w).contiguous()
        ll = self.ll_conv(x)
        high = self.high_conv(x).view(b,c,n-1,h,w).contiguous()
        x = self.enc_conv(x)
        x = self.zeroconv(x)
        return x,ll,high

class dwt_revadapter (nn.Module):
    def __init__(self, in_channels, out_channels,enc_channel, kernel_size=5, stride=1):
        super(dwt_revadapter, self).__init__()

        self.wtblock = WT_revblock(in_channels, out_channels,enc_channel, kernel_size=5, stride=1, bias=True, wt_levels=2, wt_type='db1')
        self.zeroconv = ZeroConv2d(in_channels, enc_channel, 3,padding=1, stride=stride)

    def forward(self, ll,high,enc):
        ll = self.wtblock(ll,high,enc)
        add = self.zeroconv(ll)
        return add,ll

                
class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels,dwt_dim = 16 ,double_z=True, use_linear_attn=False, attn_type="vanilla",res_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        self.conv_in2 = torch.nn.Conv2d(in_channels,
                                       dwt_dim,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult

        self.adapters  = nn.ModuleList()

        self.down = nn.ModuleList()
        
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(make_res(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,res_type=res_type))
                
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            
            down = nn.Module()
            down.block = block
            down.attn = attn
            
            # down.adapters = adapters  # Add adapters to down
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2

                self.adapters.append(dwt_adapter(dwt_dim,dwt_dim,block_out))
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = make_res(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,res_type=res_type)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = make_res(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,res_type=res_type)
        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2 * z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

        

    def forward(self, x):
        # timestep embedding
        temb = None
       

        # downsampling
        hs = [self.conv_in(x)]
        adapter_in =[self.conv_in2(x)]
        high_list = []
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                
                # h = h + self.down[i_level].adapters[i_block](torch.cat([adapter_in[i_level],h], dim=1))  # Apply Adapter
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(hs[-1])
                add,ll,high  = self.adapters[i_level](adapter_in[i_level],h)
                h = h + add
                adapter_in.append(ll)
                high_list.append(high)
                hs.append(h)

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h,ll,high_list


