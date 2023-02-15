import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

import math


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        """time embedding"""
        super().__init__()
        assert d_model % 2 == 0
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.embedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.initialize()
    
    def initialize(self):
        """initialize weight & bias"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        """[B] -> [B, C]"""
        emb = self.embedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch: int):
        """unet downsample"""
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()
    
    def initialize(self):
        """initialize weight & bias"""
        init.xavier_uniform_(self.conv.weight)
        init.zeros_(self.conv.bias)

    def forward(self, x, temb):
        """[B, C, H, W] -> [B, C, H // 2, W // 2]"""
        assert len(x.shape) == 4
        x = self.conv(x)
        return x
        

class UpSample(nn.Module):
    def __init__(self, in_ch: int):
        """unet upsample"""
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()
    
    def initialize(self):
        """initialize weight & bias"""
        init.xavier_uniform_(self.conv.weight)
        init.zeros_(self.conv.bias)
    
    def forward(self, x, temb):
        """[B, C, H, W] -> [B, C, H * 2, W * 2]"""
        assert len(x.shape) == 4
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x


class Attention(nn.Module):
    def __init__(self, in_ch):
        """self attention block"""
        super().__init__()
        self.norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()
    
    def initialize(self):
        """initialize weight & bias"""
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)
    
    def forward(self, x):
        """self-attention dim unchange"""
        assert len(x.shape) == 4
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h
    

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        """residual block"""
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.tproj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = Attention(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()
    
    def initialize(self):
        """initialize conv & linear's weight & bias"""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        """residual block, dim unchange"""
        h = self.block1(x)
        h += self.tproj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h
    

class UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
        """unet"""
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), "attn index out of bound"
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        self.head = nn.Conv2d(3, ch, 3, stride=1, padding=1)
        self.downs = nn.ModuleList()
        chs = [ch]
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResBlock(now_ch, out_ch, tdim, dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downs.append(DownSample(now_ch))
                chs.append(now_ch)
        
        self.middles = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.ups.append(ResBlock(chs.pop() + now_ch, out_ch, tdim, dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.ups.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            nn.SiLU(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1),
        )
    
    def initialize(self):
        """initialize weight & bias"""
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)
    
    def forward(self, x, t):
        """unet, dim unchange"""
        # time embedding
        temb = self.time_embedding(t)
        # downsample
        h = self.head(x)
        hs = [h]
        for layer in self.downs:
            h = layer(h, temb)
            hs.append(h)
        # middle
        for layer in self.middles:
            h = layer(h, temb)
        #upsample
        for layer in self.ups:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)

        assert len(hs) == 0
        return h
