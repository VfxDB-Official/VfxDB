# vfx_model.py
import torch
import torch.nn as nn
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
import torch.nn.functional as F
import math
# 复用你原来的 block 定义 (ResBlock3D, DownBlock3D, UpBlock3D, timestep_embedding)
# 为了节省篇幅，这里假设你已经把 unet3d_multi_ch.py 里的辅助类（ResBlock3D 等）复制过来了
# 或者直接 from unet3d_multi_ch import ResBlock3D, DownBlock3D, UpBlock3D, timestep_embedding

# 注意：为了完整运行，你需要把 unet3d_multi_ch.py 里的 ResBlock3D, DownBlock3D, UpBlock3D 贴在这里
# 或者确保这个文件能 import 到它们。

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    device = timesteps.device
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=device) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ResBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, groups: int = 8):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = nn.GroupNorm(num_groups=min(groups, in_ch), num_channels=in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )

        self.norm2 = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)

        if in_ch != out_ch:
            self.skip = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = nn.Identity()

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        temb = self.time_mlp(t_emb)
        h = h + temb[:, :, None, None, None]

        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)

        return h + self.skip(x)


class DownBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, add_downsample: bool):
        super().__init__()
        self.res1 = ResBlock3D(in_ch, out_ch, time_emb_dim)
        self.res2 = ResBlock3D(out_ch, out_ch, time_emb_dim)
        self.add_downsample = add_downsample
        if add_downsample:
            self.down = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)
        else:
            self.down = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        h = self.res1(x, t_emb)
        h = self.res2(h, t_emb)
        skip = h
        h = self.down(h)
        return h, skip


class UpBlock3D(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.conv_up = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.res1 = ResBlock3D(out_ch + skip_ch, out_ch, time_emb_dim)
        self.res2 = ResBlock3D(out_ch, out_ch, time_emb_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor):
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        x = self.conv_up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        return x

class UNet3DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4, 4),
        time_emb_dim: int = 256,
        out_channels: int = 1,
        num_classes: int = 0,
        sample_size: int = 32, # 新增：用于 config 记录 input 尺寸
        vfxdb_inference_meta: dict = None,
    ):
        super().__init__()

        # --- 原有逻辑保持不变 ---
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.time_emb_dim = time_emb_dim
        self.num_classes = int(num_classes)
        self._occ_aux_logits = None

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        if self.num_classes > 0:
            self.class_embed = nn.Embedding(self.num_classes + 1, self.time_emb_dim)

        # Down Path
        self.downs = nn.ModuleList()
        ch = in_channels
        self.down_channels = []
        num_levels = len(channel_mults)
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            add_down = i < (num_levels - 1)
            # 假设 DownBlock3D 已经定义
            self.downs.append(DownBlock3D(ch, out_ch, time_emb_dim, add_downsample=add_down))
            self.down_channels.append(out_ch)
            ch = out_ch

        # Mid
        self.mid1 = ResBlock3D(ch, ch, time_emb_dim)
        self.mid2 = ResBlock3D(ch, ch, time_emb_dim)

        # Up Path
        self.ups = nn.ModuleList()
        for i in reversed(range(num_levels - 1)):
            skip_ch = self.down_channels[i]
            out_ch = skip_ch
            self.ups.append(UpBlock3D(ch, skip_ch=skip_ch, out_ch=out_ch, time_emb_dim=time_emb_dim))
            ch = out_ch

        # Out
        self.out_norm = nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv3d(ch, out_channels, kernel_size=3, padding=1)

        # Aux Heads
        self.occ_aux_heads = nn.ModuleList()
        for i in range(num_levels - 1):
            self.occ_aux_heads.append(nn.Conv3d(self.down_channels[i], 1, kernel_size=1))

    @property
    def uncond_id(self) -> int:
        return int(self.num_classes)

    def get_occ_aux_logits(self):
        return self._occ_aux_logits

    # 稍微修改 forward 签名以适配标准调用习惯 (sample, timestep, class_labels)
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        class_labels: torch.Tensor = None,
        return_dict: bool = True
    ) -> torch.Tensor:
        # 兼容性处理
        x = sample
        timesteps = timestep
        y = class_labels

        self._occ_aux_logits = None
        aux_logits = []

        # --- Embedding ---
        t_emb = timestep_embedding(timesteps, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        if self.num_classes > 0:
            if y is None:
                # 自动处理 unconditional
                y = torch.full((x.shape[0],), self.uncond_id, device=x.device, dtype=torch.long)
            t_emb = t_emb + self.class_embed(y)

        # --- UNet Core ---
        h = x
        skips = []
        num_levels = len(self.downs)

        for i, down in enumerate(self.downs):
            h, skip = down(h, t_emb)
            if i < num_levels - 1:
                skips.append(skip)
                if self.out_channels >= 2:
                    aux_logits.append(self.occ_aux_heads[i](skip))

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        for up in self.ups:
            skip = skips.pop()
            h = up(h, skip, t_emb)

        h = self.out_norm(h)
        h = self.out_act(h)
        out = self.out_conv(h)

        if self.out_channels >= 2:
            self._occ_aux_logits = aux_logits

        if not return_dict:
            return (out,)

        return {
            "sample": out,
            "occ_aux_logits": self._occ_aux_logits
        }
