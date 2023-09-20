import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from modules.attention.base_attention import Attention
from modules.conv.base_conv import conform_conv


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)

        return out * gate.sigmoid()
        # return torch.tanh(out) * gate.sigmoid()


class downblock(nn.Module):
    def __init__(self, down, indim, outdim):
        super().__init__()
        self.c = nn.Conv1d(indim, outdim * 2, kernel_size=down * 2, stride=down, padding=down // 2)
        self.act = GLU(1)
        self.out = nn.Conv1d(outdim, outdim, kernel_size=3, padding=1)
        self.act1 = nn.GELU()

    def forward(self, x):
        return self.act1(self.out(self.act(self.c(x))))


class upblock(nn.Module):
    def __init__(self, ups, indim, outdim):
        super().__init__()
        self.c = nn.ConvTranspose1d(indim, outdim * 2, kernel_size=ups * 2, stride=ups, padding=ups // 2)
        self.act = GLU(1)
        self.out = nn.Conv1d(outdim, outdim, kernel_size=3, padding=1)
        self.act1 = nn.GELU()

    def forward(self, x):
        return self.act1(self.out(self.act(self.c(x))))


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):

    def __init__(self, dim, dim_out, time_emb_dim=None, groups=32, ):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if time_emb_dim is not None
            else None
        )
        # self.mlp2 = (
        #     nn.Sequential(nn.SiLU(), nn.Conv1d(dim_out, dim_out * 2, kernel_size=1))
        #     if time_emb_dim is not None
        #     else None
        # )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, ):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c  1")
            # if cs is not  None:
            #     csx = self.mlp2(cs)
            #     time_emb = csx+time_emb
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class unet_conformer_full(nn.Module):
    def __init__(self, downs=[2, 2, 2], dim=[512, 768, 1024], latentdim=1024, indim=128, outdim=512,
                 cf_kernel_size=31, cf_conv_drop=0.1, attention_drop=0.1, attention_heads_dim=64):
        super().__init__()
        self.incon = nn.Conv1d(indim, dim[0], 1, padding=0)
        self.outcon = nn.Conv1d(dim[0], outdim, 1, padding=0) if dim[0] != outdim else nn.Identity()
        # self.outcon1 = nn.Conv1d(dim[0], dim[0], 1, padding=0)
        self.dlres = nn.ModuleList()
        self.dlatt = nn.ModuleList()
        self.dldd = nn.ModuleList()

        # self.mlp = (
        #     nn.Sequential(nn.SiLU(), nn.Linear(512, 256))
        #     # if time_emb_dim is not None
        #     # else None
        # )

        dims = dim.copy()
        dims.append(latentdim)
        igx = 1

        for idx, i in enumerate(downs):
            self.dlres.append(nn.ModuleList([ResnetBlock(dim[idx], dim[idx], groups=32),
                                             ResnetBlock(dim[idx], dim[idx], groups=32)]))
            h = dim[idx] // attention_heads_dim
            self.dlatt.append(
                nn.ModuleList([cf_pack(lay=1, dim=dim[idx], kernel_size=cf_kernel_size, conv_drop=cf_conv_drop,
                                       attention_drop=attention_drop, attention_heads=h,
                                       attention_heads_dim=attention_heads_dim),
                               cf_pack(lay=1, dim=dim[idx], kernel_size=cf_kernel_size, conv_drop=cf_conv_drop,
                                       attention_drop=attention_drop, attention_heads=h,
                                       attention_heads_dim=attention_heads_dim)]))
            self.dldd.append(downblock(i, dims[idx], dims[idx + 1]))

            igx = igx * i

        # self.csds = nn.Conv1d(cdim, dim[0], kernel_size=1)

        self.up_samples_res = nn.ModuleList()
        self.upatt = nn.ModuleList()
        self.updd = nn.ModuleList()

        ups = downs.copy()
        ups.reverse()
        upsd = dim.copy()
        upsd.reverse()

        upsds = dims.copy()
        upsds.reverse()

        for idx, i in enumerate(ups):
            self.up_samples_res.append(nn.ModuleList([ResnetBlock(upsd[idx] * 2, upsd[idx], groups=32),
                                                      ResnetBlock(upsd[idx] * 2, upsd[idx], groups=32)]))
            h = dim[idx] // attention_heads_dim
            self.upatt.append(
                nn.ModuleList([cf_pack(lay=1, dim=upsd[idx], kernel_size=cf_kernel_size, conv_drop=cf_conv_drop,
                                       attention_drop=attention_drop, attention_heads=h,
                                       attention_heads_dim=attention_heads_dim),
                               cf_pack(lay=1, dim=upsd[idx], kernel_size=cf_kernel_size, conv_drop=cf_conv_drop,
                                       attention_drop=attention_drop, attention_heads=h,
                                       attention_heads_dim=attention_heads_dim)]))
            self.updd.append(upblock(i, upsds[idx], upsds[idx + 1]))

        self.mres1 = ResnetBlock(latentdim, latentdim, groups=32)
        # self.matt = PreNorm(latentdim,LinearAttention(latentdim,heads=32))
        # self.matt = attLp(latentdim, heads=16, cdim=cdim)

        self.matt = cf_pack(lay=2, dim=latentdim, kernel_size=cf_kernel_size, conv_drop=cf_conv_drop,
                            attention_drop=attention_drop, attention_heads=latentdim // attention_heads_dim,
                            attention_heads_dim=attention_heads_dim)

        self.mres2 = ResnetBlock(latentdim, latentdim, groups=32)

        self.outres = nn.ModuleList([ResnetBlock(dim[0] * 2, dim[0], groups=32), ])
        # self.outatt = attLp(dim[0], heads=dim[0] // 32, cdim=cdim)

        self.outatt = cf_pack(lay=1, dim=dim[0], kernel_size=cf_kernel_size, conv_drop=cf_conv_drop,
                              attention_drop=attention_drop, attention_heads=dim[0] // attention_heads_dim,
                              attention_heads_dim=attention_heads_dim)

    def forward(self, x, ):
        fff = []
        # x=torch.cat([x, cs], dim=1)
        # tpp=self.mlp(time_emb)
        # tpp = rearrange(tpp, "b c -> b c  1")
        # csa=cs

        x = self.incon(x)
        # x = F.relu(x)
        res11 = x.clone()

        for res, att, don, in zip(self.dlres, self.dlatt, self.dldd, ):
            cccc = []

            for idx, ii in enumerate(res):
                # x=torch.cat([x,csdxd], dim=1)

                x = ii(x, )
                x = att[idx](x, )
                cccc.append(x)
            # x=att[1](x,cs)
            # cccc[1]=x
            fff.append(cccc)
            x = don(x)
        x = self.mres2(self.matt(self.mres1(x, ), ), )
        fff.reverse()

        for res, att, up, fautres, in zip(self.up_samples_res, self.upatt, self.updd, fff, ):
            x = up(x)
            # x=fautres+x

            for idx, ii in enumerate(res):
                x = torch.cat([x, fautres[idx]], dim=1)
                x = ii(x, )
                x = att[idx](x, )

        x = torch.cat([x, res11, ], dim=1)
        for ii in self.outres:
            x = ii(x, )
        x = self.outatt(x, )
        # x = F.relu(self.outcon1(x))
        return self.outcon(x)


class unet_conformer(nn.Module):
    def __init__(self, downs=[2, 2, 2], dim=[512, 768, 1024], latentdim=1024, indim=128, outdim=512,
                 cf_kernel_size=31, cf_conv_drop=0.1, attention_drop=0.1, attention_heads_dim=64):
        super().__init__()
        self.incon = nn.Conv1d(indim, dim[0], 1, padding=0)
        self.outcon = nn.Conv1d(dim[0], outdim, 1, padding=0) if dim[0] != outdim else nn.Identity()
        # self.outcon1 = nn.Conv1d(dim[0], dim[0], 1, padding=0)
        self.dlres = nn.ModuleList()
        # self.dlatt = nn.ModuleList()
        self.dldd = nn.ModuleList()

        # self.mlp = (
        #     nn.Sequential(nn.SiLU(), nn.Linear(512, 256))
        #     # if time_emb_dim is not None
        #     # else None
        # )

        dims = dim.copy()
        dims.append(latentdim)

        for idx, i in enumerate(downs):
            self.dlres.append(nn.ModuleList([ResnetBlock(dim[idx], dim[idx], groups=32),
                                             ResnetBlock(dim[idx], dim[idx], groups=32)]))

            self.dldd.append(downblock(i, dims[idx], dims[idx + 1]))

        # self.csds = nn.Conv1d(cdim, dim[0], kernel_size=1)

        self.up_samples_res = nn.ModuleList()
        # self.upatt = nn.ModuleList()
        self.updd = nn.ModuleList()

        ups = downs.copy()
        ups.reverse()
        upsd = dim.copy()
        upsd.reverse()

        upsds = dims.copy()
        upsds.reverse()

        for idx, i in enumerate(ups):
            self.up_samples_res.append(nn.ModuleList([ResnetBlock(upsd[idx] * 2, upsd[idx], groups=32),
                                                      ResnetBlock(upsd[idx] * 2, upsd[idx], groups=32)]))
            h = dim[idx] // attention_heads_dim

            self.updd.append(upblock(i, upsds[idx], upsds[idx + 1]))

        self.mres1 = ResnetBlock(latentdim, latentdim, groups=32)
        # self.matt = PreNorm(latentdim,LinearAttention(latentdim,heads=32))
        # self.matt = attLp(latentdim, heads=16, cdim=cdim)

        self.matt = cf_pack(lay=2, dim=latentdim, kernel_size=cf_kernel_size, conv_drop=cf_conv_drop,
                            attention_drop=attention_drop, attention_heads=latentdim // attention_heads_dim,
                            attention_heads_dim=attention_heads_dim)

        self.mres2 = ResnetBlock(latentdim, latentdim, groups=32)

        self.outres = nn.ModuleList([ResnetBlock(dim[0] * 2, dim[0], groups=32), ])
        # self.outatt = attLp(dim[0], heads=dim[0] // 32, cdim=cdim)

    def forward(self, x, ):
        fff = []
        # x=torch.cat([x, cs], dim=1)
        # tpp=self.mlp(time_emb)
        # tpp = rearrange(tpp, "b c -> b c  1")
        # csa=cs

        x = self.incon(x)
        # x = F.relu(x)
        res11 = x.clone()

        for res, don, in zip(self.dlres, self.dldd, ):
            cccc = []

            for idx, ii in enumerate(res):
                # x=torch.cat([x,csdxd], dim=1)

                x = ii(x, )

                cccc.append(x)
            # x=att[1](x,cs)
            # cccc[1]=x
            fff.append(cccc)
            x = don(x)
        x = self.mres2(self.matt(self.mres1(x, ), ), )
        fff.reverse()

        for res, up, fautres, in zip(self.up_samples_res, self.updd, fff, ):
            x = up(x)
            # x=fautres+x

            for idx, ii in enumerate(res):
                x = torch.cat([x, fautres[idx]], dim=1)
                x = ii(x, )

        x = torch.cat([x, res11, ], dim=1)
        for ii in self.outres:
            x = ii(x, )
        #
        # x = F.relu(self.outcon1(x))
        return self.outcon(x)


class unet(nn.Module):
    def __init__(self, downs=[2, 2, 2], dim=[512, 768, 1024], latentdim=1024, indim=128, outdim=512,
                 ):
        super().__init__()
        self.incon = nn.Conv1d(indim, dim[0], 1, padding=0)
        self.outcon = nn.Conv1d(dim[0], outdim, 1, padding=0) if dim[0] != outdim else nn.Identity()
        # self.outcon1 = nn.Conv1d(dim[0], dim[0], 1, padding=0)
        self.dlres = nn.ModuleList()
        # self.dlatt = nn.ModuleList()
        self.dldd = nn.ModuleList()

        # self.mlp = (
        #     nn.Sequential(nn.SiLU(), nn.Linear(512, 256))
        #     # if time_emb_dim is not None
        #     # else None
        # )

        dims = dim.copy()
        dims.append(latentdim)

        for idx, i in enumerate(downs):
            self.dlres.append(nn.ModuleList([ResnetBlock(dim[idx], dim[idx], groups=32),
                                             ResnetBlock(dim[idx], dim[idx], groups=32)]))

            self.dldd.append(downblock(i, dims[idx], dims[idx + 1]))

        # self.csds = nn.Conv1d(cdim, dim[0], kernel_size=1)

        self.up_samples_res = nn.ModuleList()
        # self.upatt = nn.ModuleList()
        self.updd = nn.ModuleList()

        ups = downs.copy()
        ups.reverse()
        upsd = dim.copy()
        upsd.reverse()

        upsds = dims.copy()
        upsds.reverse()

        for idx, i in enumerate(ups):
            self.up_samples_res.append(nn.ModuleList([ResnetBlock(upsd[idx] * 2, upsd[idx], groups=32),
                                                      ResnetBlock(upsd[idx] * 2, upsd[idx], groups=32)]))

            self.updd.append(upblock(i, upsds[idx], upsds[idx + 1]))

        self.mres1 = ResnetBlock(latentdim, latentdim, groups=32)
        # self.matt = PreNorm(latentdim,LinearAttention(latentdim,heads=32))
        # self.matt = attLp(latentdim, heads=16, cdim=cdim)

        # self.matt = cf_pack(lay=2, dim=latentdim, kernel_size=cf_kernel_size, conv_drop=cf_conv_drop,
        #                     attention_drop=attention_drop, attention_heads=latentdim // attention_heads_dim,
        #                     attention_heads_dim=attention_heads_dim)

        self.mres2 = ResnetBlock(latentdim, latentdim, groups=32)

        self.outres = nn.ModuleList([ResnetBlock(dim[0] * 2, dim[0], groups=32), ])
        # self.outatt = attLp(dim[0], heads=dim[0] // 32, cdim=cdim)

    def forward(self, x, ):
        fff = []
        # x=torch.cat([x, cs], dim=1)
        # tpp=self.mlp(time_emb)
        # tpp = rearrange(tpp, "b c -> b c  1")
        # csa=cs

        x = self.incon(x)
        # x = F.relu(x)
        res11 = x.clone()

        for res, don, in zip(self.dlres, self.dldd, ):
            cccc = []

            for idx, ii in enumerate(res):
                # x=torch.cat([x,csdxd], dim=1)

                x = ii(x, )

                cccc.append(x)
            # x=att[1](x,cs)
            # cccc[1]=x
            fff.append(cccc)
            x = don(x)
        x = self.mres2(self.mres1(x, ), )
        fff.reverse()

        for res, up, fautres, in zip(self.up_samples_res, self.updd, fff, ):
            x = up(x)
            # x=fautres+x

            for idx, ii in enumerate(res):
                x = torch.cat([x, fautres[idx]], dim=1)
                x = ii(x, )

        x = torch.cat([x, res11, ], dim=1)
        for ii in self.outres:
            x = ii(x, )
        #
        # x = F.relu(self.outcon1(x))
        return self.outcon(x)


class conform_ffn(nn.Module):
    def __init__(self, dim, DropoutL1: float = 0.1, DropoutL2: float = 0.1):
        super().__init__()
        self.ln1 = nn.Linear(dim, dim * 4)
        self.ln2 = nn.Linear(dim * 4, dim)
        self.drop1 = nn.Dropout(DropoutL1) if DropoutL1 > 0. else nn.Identity()
        self.drop2 = nn.Dropout(DropoutL2) if DropoutL2 > 0. else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.ln1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.ln2(x)
        return self.drop2(x)


class conform_blocke(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 31, conv_drop: float = 0.1, ffn_latent_drop: float = 0.1,
                 ffn_out_drop: float = 0.1, attention_drop: float = 0.1, attention_heads: int = 4,
                 attention_heads_dim: int = 64):
        super().__init__()
        self.ffn1 = conform_ffn(dim, ffn_latent_drop, ffn_out_drop)
        self.ffn2 = conform_ffn(dim, ffn_latent_drop, ffn_out_drop)
        self.att = Attention(dim, heads=attention_heads, dim_head=attention_heads_dim)
        self.attdrop = nn.Dropout(attention_drop) if attention_drop > 0. else nn.Identity()
        self.conv = conform_conv(dim, kernel_size=kernel_size,

                                 DropoutL=conv_drop, )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.norm5 = nn.LayerNorm(dim)

    def forward(self, x, mask=None, ):
        x = self.ffn1(self.norm1(x)) * 0.5 + x

        x = self.attdrop(self.att(self.norm2(x), mask=mask)) + x
        x = self.conv(self.norm3(x)) + x
        x = self.ffn2(self.norm4(x)) * 0.5 + x
        return self.norm5(x)


class light_conform_blocke(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 31, conv_drop: float = 0.1, attention_drop: float = 0.1,
                 attention_heads: int = 4,
                 attention_heads_dim: int = 64):
        super().__init__()

        self.att = Attention(dim, heads=attention_heads, dim_head=attention_heads_dim)
        self.attdrop = nn.Dropout(attention_drop) if attention_drop > 0. else nn.Identity()
        self.conv = conform_conv(dim, kernel_size=kernel_size,

                                 DropoutL=conv_drop, )

        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.norm5 = nn.LayerNorm(dim)

    def forward(self, x, mask=None, ):
        x = self.attdrop(self.att(self.norm2(x), mask=mask)) + x
        x = self.conv(self.norm3(x)) + x

        return self.norm5(x)


class cf_pack(nn.Module):
    def __init__(self, lay: int, dim: int, kernel_size: int = 31, conv_drop: float = 0.1, attention_drop: float = 0.1,
                 attention_heads: int = 4,
                 attention_heads_dim: int = 64):
        super().__init__()
        self.net = nn.ModuleList([light_conform_blocke(dim=dim, kernel_size=kernel_size, conv_drop=conv_drop,
                                                       attention_drop=attention_drop, attention_heads=attention_heads,
                                                       attention_heads_dim=attention_heads_dim) for _ in range(lay)])

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        x = x.transpose(1, 2)
        for i in self.net:
            x = i(x, mask)
        return x.transpose(1, 2)


class base_cf_pack(nn.Module):
    def __init__(self, lay: int, dim: int, kernel_size: int = 31, conv_drop: float = 0.1, attention_drop: float = 0.1,
                 attention_heads: int = 4,
                 attention_heads_dim: int = 64, ffn_latent_drop=0.1, ffn_out_drop=0.1):
        super().__init__()
        self.net = nn.ModuleList([conform_blocke(dim=dim, kernel_size=kernel_size, conv_drop=conv_drop,
                                                 attention_drop=attention_drop, attention_heads=attention_heads,
                                                 attention_heads_dim=attention_heads_dim,ffn_latent_drop=ffn_latent_drop,ffn_out_drop=ffn_out_drop) for _ in range(lay)])

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:

        for i in self.net:
            x = i(x, mask)
        return x


class unet_adp(nn.Module):
    def __init__(self, unet_type='cf_unet_full', unet_down=[2, 2, 2], unet_dim=[512, 768, 1024], unet_latentdim=1024,
                 unet_indim=128, unet_outdim=512,
                 unet_cf_kernel_size=31, unet_cf_conv_drop=0.1, unet_attention_drop=0.1, unet_attention_heads_dim=64):
        super().__init__()
        self.downx = None
        for i in unet_down:
            if self.downx is None:
                self.downx = unet_down[0]
            else:
                self.downx = self.downx * i
        if unet_type == 'cf_unet_full':
            self.unet = unet_conformer_full(downs=unet_down, dim=unet_dim, latentdim=unet_latentdim, indim=unet_indim,
                                            outdim=unet_outdim,
                                            cf_kernel_size=unet_cf_kernel_size, cf_conv_drop=unet_cf_conv_drop,
                                            attention_drop=unet_attention_drop,
                                            attention_heads_dim=unet_attention_heads_dim)
        elif unet_type == 'cf_unet':
            self.unet = unet_conformer(downs=unet_down, dim=unet_dim, latentdim=unet_latentdim, indim=unet_indim,
                                       outdim=unet_outdim,
                                       cf_kernel_size=unet_cf_kernel_size, cf_conv_drop=unet_cf_conv_drop,
                                       attention_drop=unet_attention_drop,
                                       attention_heads_dim=unet_attention_heads_dim)
        elif unet_type == 'unet':
            self.unet = unet(downs=unet_down, dim=unet_dim, latentdim=unet_latentdim, indim=unet_indim,
                             outdim=unet_outdim,
                             )
        else:
            raise RuntimeError("unsupoort")

    def forward(self, x):
        x = x.transpose(1, 2)
        b1, c1, t1 = x.shape

        pad = t1 % self.downx
        if pad != 0:
            pad = self.downx - pad
        x = F.pad(x, (0, pad), "constant", 0)

        x = self.unet(x, )
        if pad != 0:
            x = x[:, :, :-pad]
        return x.transpose(1, 2)


class unet_base_cf(nn.Module):
    def __init__(self, output_lay: int, dim: int, indim: int, outdim: int,  kernel_size: int = 31,
                 conv_drop: float = 0.1,
                 ffn_latent_drop: float = 0.1,
                 ffn_out_drop: float = 0.1, attention_drop: float = 0.1, attention_heads: int = 4,
                 attention_heads_dim: int = 64,sig:bool=True,unet_type='cf_unet_full',unet_down=[2, 2, 2], unet_dim=[512, 768, 1024], unet_latentdim=1024,):
        super().__init__()
        self.sig=sig

        self.unet=unet_adp(unet_type=unet_type, unet_down=unet_down, unet_dim=unet_dim, unet_latentdim=unet_latentdim,
                 unet_indim=indim, unet_outdim=dim,
                 unet_cf_kernel_size=kernel_size, unet_cf_conv_drop=conv_drop, unet_attention_drop=attention_drop, unet_attention_heads_dim=attention_heads_dim)
        self.outln = nn.Linear(dim, outdim)
        self.cutheard = nn.Linear(dim, 1)
        # self.cutheard = nn.Linear(dim, outdim)



        self.att1=base_cf_pack(lay=output_lay,dim=dim, kernel_size=kernel_size, conv_drop=conv_drop, ffn_latent_drop=ffn_latent_drop,
                            ffn_out_drop=ffn_out_drop, attention_drop=attention_drop, attention_heads=attention_heads,
                            attention_heads_dim=attention_heads_dim)
        self.att2 = base_cf_pack(lay=output_lay,dim=dim, kernel_size=kernel_size, conv_drop=conv_drop, ffn_latent_drop=ffn_latent_drop,
                            ffn_out_drop=ffn_out_drop, attention_drop=attention_drop, attention_heads=attention_heads,
                            attention_heads_dim=attention_heads_dim)


    def forward(self, x, pitch=None, mask=None):

        # torch.masked_fill()
        x=self.unet(x)



        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)

        xo=self.att1(x)
        x1 = self.att2(x)



        if mask is not None:
            xo = xo.masked_fill(~mask.unsqueeze(-1), 0)
            x1 = x1.masked_fill(~mask.unsqueeze(-1), 0)


        cutprp = self.cutheard(x1)
        midiout = self.outln(xo)
        cutprp = torch.sigmoid(cutprp)
        cutprp = torch.squeeze(cutprp, -1)
        # if  self.sig:
        #     midiout = torch.sigmoid(midiout)
        return midiout, cutprp



if __name__ == '__main__':
    fff = unet_base_cf( dim=512,indim=128,outdim=256,output_lay=1,unet_down=[2, 2, 4], unet_dim=[128, 128, 128], unet_latentdim=128)
    aaa = fff(torch.randn(2, 255, 128))
    pass
