import torch
import torch.nn as nn
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import Tuple, Optional, List
import logging
from operator import mul
from functools import reduce

@dataclass
class MultiScaleVitCfg:
    depths: Tuple[int, ...] = (1, 2, 11, 2)
    embed_dim: Tuple[int, ...] = None
    num_heads: Tuple[int, ...] = (4, 4, 8, 8)
    mlp_ratio: float = 4.0
    pool_first: bool = False
    expand_attn: bool = True
    qkv_bias: bool = True
    use_cls_token: bool = False
    use_abs_pos: bool = True
    residual_pooling: bool = True
    mode: str = 'conv'
    kernel_qkv: Tuple[int, int] = (3, 3)
    stride_q: Tuple[Tuple[int, int]] = ((1, 1), (2, 2), (2, 2), (2, 2))
    stride_kv: Optional[Tuple[Tuple[int, int]]] = None
    stride_kv_adaptive: Tuple[int, int] = (4, 4)
    norm_layer: str = 'layernorm'
    norm_eps: float = 1e-6

    def __post_init__(self):
        num_stages = len(self.depths)
        if self.embed_dim is None:
            self.embed_dim = tuple(self.embed_dim[0] * (2 ** i) for i in range(num_stages))
        if not isinstance(self.num_heads, (tuple, list)):
            self.num_heads = tuple(self.num_heads * 2 ** i for i in range(num_stages))
        assert len(self.num_heads) == num_stages
        if self.stride_kv_adaptive is not None and self.stride_kv is None:
            _stride_kv = self.stride_kv_adaptive
            pool_kv_stride = []
            for i in range(num_stages):
                if min(self.stride_q[i]) > 1:
                    _stride_kv = [
                        max(_stride_kv[d] // self.stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                pool_kv_stride.append(tuple(_stride_kv))
            self.stride_kv = tuple(pool_kv_stride)

def prod(iterable):
    return reduce(mul, iterable, 1)

def trunc_normal_tf_(tensor, std=0.02):
    nn.init.trunc_normal_(tensor, mean=0.0, std=std)

def get_norm_layer(norm_type):
    if norm_type == 'layernorm':
        return nn.LayerNorm
    raise NotImplementedError(f"Unsupported norm layer: {norm_type}")

def reshape_pre_pool(x, feat_size: Tuple[int, int], has_cls_token: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    H, W = feat_size
    if has_cls_token:
        cls_tok, x = x[:, :, :1, :], x[:, :, 1:, :]
    else:
        cls_tok = None
    x = x.reshape(-1, H, W, x.shape[-1]).permute(0, 3, 1, 2).contiguous()
    return x, cls_tok

def reshape_post_pool(x, num_heads: int, cls_tok: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[int, int]]:
    feat_size = (x.shape[2], x.shape[3])
    L_pooled = x.shape[2] * x.shape[3]
    x = x.reshape(-1, num_heads, x.shape[1], L_pooled).transpose(2, 3)
    if cls_tok is not None:
        x = torch.cat((cls_tok, x), dim=2)
    return x, feat_size

def cal_rel_pos_type(
    attn: torch.Tensor,
    q: torch.Tensor,
    has_cls_token: bool,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
):
    sp_idx = 1 if has_cls_token else 0
    q_h, q_w = q_size
    k_h, k_w = k_size

    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
        torch.arange(q_h, device=q.device).unsqueeze(-1) * q_h_ratio -
        torch.arange(k_h, device=q.device).unsqueeze(0) * k_h_ratio
    )
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
        torch.arange(q_w, device=q.device).unsqueeze(-1) * q_w_ratio -
        torch.arange(k_w, device=q.device).unsqueeze(0) * k_w_ratio
    )
    dist_w += (k_w - 1) * k_w_ratio

    rel_h = rel_pos_h[dist_h.long()]
    rel_w = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, rel_h)
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, rel_w)

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w)
        + rel_h.unsqueeze(-1)
        + rel_w.unsqueeze(-2)
    ).view(B, -1, q_h * q_w, k_h * k_w)

    return attn

class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        feat_size,
        num_heads=8,
        qkv_bias=True,
        mode="conv",
        kernel_q=(1, 1),
        kernel_kv=(1, 1),
        stride_q=(1, 1),
        stride_kv=(1, 1),
        has_cls_token=False,
        rel_pos_type='spatial',
        residual_pooling=True,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim_out = dim_out
        self.head_dim = dim_out // num_heads
        self.scale = self.head_dim ** -0.5
        self.has_cls_token = has_cls_token
        padding_q = tuple([int(q // 2) for q in kernel_q])
        padding_kv = tuple([int(kv // 2) for kv in kernel_kv])

        self.qkv = nn.Linear(dim, dim_out * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim_out, dim_out)

        if prod(kernel_q) == 1 and prod(stride_q) == 1:
            kernel_q = None
        if prod(kernel_kv) == 1 and prod(stride_kv) == 1:
            kernel_kv = None
        self.mode = mode
        self.unshared = mode == 'conv_unshared'
        self.norm_q, self.norm_k, self.norm_v = None, None, None
        self.pool_q, self.pool_k, self.pool_v = None, None, None
        if mode in ("avg", "max"):
            pool_op = nn.MaxPool2d if mode == "max" else nn.AvgPool2d
            if kernel_q:
                self.pool_q = pool_op(kernel_q, stride_q, padding_q)
            if kernel_kv:
                self.pool_k = pool_op(kernel_kv, stride_kv, padding_kv)
                self.pool_v = pool_op(kernel_kv, stride_kv, padding_kv)
        elif mode == "conv" or mode == "conv_unshared":
            dim_conv = dim_out // num_heads if mode == "conv" else dim_out
            if kernel_q:
                self.pool_q = nn.Conv2d(
                    dim_conv,
                    dim_conv,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=dim_conv,
                    bias=False,
                )
                self.norm_q = norm_layer(dim_conv)
            if kernel_kv:
                self.pool_k = nn.Conv2d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                self.norm_k = norm_layer(dim_conv)
                self.pool_v = nn.Conv2d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                self.norm_v = norm_layer(dim_conv)

        self.rel_pos_type = rel_pos_type
        if self.rel_pos_type == 'spatial':
            q_size = max(feat_size[0] // stride_q[0], 1) if len(stride_q) > 0 else feat_size[0]
            kv_size = max(feat_size[0] // stride_kv[0], 1) if len(stride_kv) > 0 else feat_size[0]
            rel_sp_dim = 2 * max(q_size, kv_size) - 1
            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, self.head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, self.head_dim))
            trunc_normal_tf_(self.rel_pos_h, std=0.02)
            trunc_normal_tf_(self.rel_pos_w, std=0.02)

        self.residual_pooling = residual_pooling

    def forward(self, x, feat_size: Tuple[int, int]):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        if self.pool_q is not None:
            q, q_tok = reshape_pre_pool(q, feat_size, self.has_cls_token)
            q = self.pool_q(q)
            q, q_size = reshape_post_pool(q, self.num_heads, q_tok)
        else:
            q_size = feat_size
        if self.norm_q is not None:
            q = self.norm_q(q)

        if self.pool_k is not None:
            k, k_tok = reshape_pre_pool(k, feat_size, self.has_cls_token)
            k = self.pool_k(k)
            k, k_size = reshape_post_pool(k, self.num_heads, k_tok)
        else:
            k_size = feat_size
        if self.norm_k is not None:
            k = self.norm_k(k)

        if self.pool_v is not None:
            v, v_tok = reshape_pre_pool(v, feat_size, self.has_cls_token)
            v = self.pool_v(v)
            v, _ = reshape_post_pool(v, self.num_heads, v_tok)
        if self.norm_v is not None:
            v = self.norm_v(v)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.rel_pos_type == 'spatial':
            attn = cal_rel_pos_type(
                attn,
                q,
                self.has_cls_token,
                q_size,
                k_size,
                self.rel_pos_h,
                self.rel_pos_w,
            )
        attn = attn.softmax(dim=-1)
        x = attn @ v

        if self.residual_pooling:
            x = x + q

        x = x.transpose(1, 2).reshape(B, -1, self.dim_out)
        x = self.proj(x)
        return x, q_size

class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        feat_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        kernel_q=(1, 1),
        kernel_kv=(1, 1),
        stride_q=(1, 1),
        stride_kv=(1, 1),
        mode="conv",
        has_cls_token=False,
        expand_attn=False,
        rel_pos_type='spatial',
        residual_pooling=True,
    ):
        super().__init__()
        proj_needed = dim != dim_out
        self.dim = dim
        self.dim_out = dim_out
        self.has_cls_token = has_cls_token

        self.norm1 = norm_layer(dim)
        self.shortcut_proj_attn = nn.Linear(dim, dim_out) if proj_needed and expand_attn else None
        if stride_q and prod(stride_q) > 1:
            kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
            stride_skip = stride_q
            padding_skip = [int(skip // 2) for skip in kernel_skip]
            self.shortcut_pool_attn = nn.MaxPool2d(kernel_skip, stride_skip, padding_skip)
        else:
            self.shortcut_pool_attn = None

        att_dim = dim_out if expand_attn else dim
        self.attn = MultiScaleAttention(
            dim,
            att_dim,
            num_heads=num_heads,
            feat_size=feat_size,
            qkv_bias=qkv_bias,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=norm_layer,
            has_cls_token=has_cls_token,
            mode=mode,
            rel_pos_type=rel_pos_type,
            residual_pooling=residual_pooling,
        )
        self.drop_path1 = nn.Identity() if drop_path == 0.0 else nn.Dropout(drop_path)

        self.norm2 = norm_layer(att_dim)
        self.shortcut_proj_mlp = nn.Linear(dim, dim_out) if proj_needed and not expand_attn else None
        self.mlp = nn.Sequential(
            nn.Linear(att_dim, int(att_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(int(att_dim * mlp_ratio), dim_out)
        )
        self.drop_path2 = nn.Identity() if drop_path == 0.0 else nn.Dropout(drop_path)

    def _shortcut_pool(self, x, feat_size: Tuple[int, int]):
        if self.shortcut_pool_attn is None:
            return x
        if self.has_cls_token:
            cls_tok, x = x[:, :1, :], x[:, 1:, :]
        else:
            cls_tok = None
        B, L, C = x.shape
        H, W = feat_size
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = self.shortcut_pool_attn(x)
        x = x.reshape(B, C, -1).transpose(1, 2)
        if cls_tok is not None:
            x = torch.cat((cls_tok, x), dim=1)
        return x

    def forward(self, x, feat_size: Tuple[int, int]):
        x_norm = self.norm1(x)
        x_shortcut = x if self.shortcut_proj_attn is None else self.shortcut_proj_attn(x_norm)
        x_shortcut = self._shortcut_pool(x_shortcut, feat_size)
        x, feat_size_new = self.attn(x_norm, feat_size)
        x = x_shortcut + self.drop_path1(x)

        x_norm = self.norm2(x)
        x_shortcut = x if self.shortcut_proj_mlp is None else self.shortcut_proj_mlp(x_norm)
        x = x_shortcut + self.drop_path2(self.mlp(x_norm))
        return x, feat_size_new

class MultiScaleVitStage(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        depth,
        num_heads,
        feat_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        mode="conv",
        kernel_q=(1, 1),
        kernel_kv=(1, 1),
        stride_q=(1, 1),
        stride_kv=(1, 1),
        has_cls_token=False,
        expand_attn=False,
        rel_pos_type='spatial',
        residual_pooling=True,
        norm_layer=nn.LayerNorm,
        drop_path=0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        if expand_attn:
            out_dims = (dim_out,) * depth
        else:
            out_dims = (dim,) * (depth - 1) + (dim_out,)

        for i in range(depth):
            attention_block = MultiScaleBlock(
                dim=dim,
                dim_out=out_dims[i],
                num_heads=num_heads,
                feat_size=feat_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                kernel_q=kernel_q,
                kernel_kv=kernel_kv,
                stride_q=stride_q if i == 0 else (1, 1),
                stride_kv=stride_kv,
                mode=mode,
                has_cls_token=has_cls_token,
                expand_attn=expand_attn,
                rel_pos_type=rel_pos_type,
                residual_pooling=residual_pooling,
                norm_layer=norm_layer,
                drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
            )
            dim = out_dims[i]
            self.blocks.append(attention_block)
            if i == 0:
                feat_size = tuple([size // stride for size, stride in zip(feat_size, stride_q)])

        self.feat_size = feat_size

    def forward(self, x, feat_size: Tuple[int, int]):
        for blk in self.blocks:
            x, feat_size = blk(x, feat_size)
        return x, feat_size

class MViTv2(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_teeth: int = 14,
        max_stages: int = 25,
        depths: List[int] = [1, 2, 11, 2],
        num_heads: List[int] = [4, 4, 8, 8],
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.2,
        teacher_forcing: bool = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_teeth = num_teeth
        self.max_stages = max_stages
        self.teacher_forcing = teacher_forcing

        self.tooth_mapping = {
            0: '31', 1: '32', 2: '33', 3: '34', 4: '35', 5: '36', 6: '37',
            7: '41', 8: '42', 9: '43', 10: '44', 11: '45', 12: '46', 13: '47'
        }
        self.grid_mapping = torch.zeros(2, 7, dtype=torch.long)
        for idx in range(num_teeth):
            row = idx // 7
            col = idx % 7
            self.grid_mapping[row, col] = idx

        cfg = MultiScaleVitCfg(
            depths=tuple(depths),
            embed_dim=(embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 4),
            num_heads=tuple(num_heads),
            mlp_ratio=mlp_ratio,
            use_abs_pos=True,
            use_cls_token=False
        )
        norm_layer = partial(get_norm_layer(cfg.norm_layer), eps=cfg.norm_eps)
        feat_size = (2, 7)

        self.input_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_teeth, embed_dim))
        num_stages = len(cfg.depths)
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(cfg.depths)).split(cfg.depths)]
        self.stages = nn.ModuleList()
        dim = embed_dim
        for i in range(num_stages):
            stage = MultiScaleVitStage(
                dim=dim,
                dim_out=cfg.embed_dim[i],
                depth=cfg.depths[i],
                num_heads=cfg.num_heads[i],
                feat_size=feat_size,
                mlp_ratio=cfg.mlp_ratio,
                qkv_bias=cfg.qkv_bias,
                mode=cfg.mode,
                kernel_q=cfg.kernel_qkv,
                kernel_kv=cfg.kernel_qkv,
                stride_q=cfg.stride_q[i],
                stride_kv=cfg.stride_kv[i],
                has_cls_token=cfg.use_cls_token,
                expand_attn=cfg.expand_attn,
                rel_pos_type='spatial',
                residual_pooling=cfg.residual_pooling,
                norm_layer=norm_layer,
                drop_path=dpr[i],
            )
            dim = cfg.embed_dim[i]
            feat_size = stage.feat_size
            self.stages.append(stage)

        self.norm = norm_layer(dim)
        self.out_layer = nn.Linear(dim, 6)  # Output 6 transformation parameters per tooth
        trunc_normal_tf_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_tf_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, targets=None, epoch=None, total_epochs=None):
        logger = logging.getLogger('TrainLogger')
        B = x.size(0)
        # Permute input to [batch_size, embed_dim, 2, 7] for Conv2d
        x = x.permute(0, 3, 1, 2).contiguous()  # From [B, 2, 7, embed_dim] to [B, embed_dim, 2, 7]
        x = self.input_conv(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, self.embed_dim)
        x = x + self.pos_embed

        feat_size = (2, 7)
        if self.training and self.teacher_forcing and targets is not None:
            alpha = min(1.0, epoch / (total_epochs * 0.75)) if epoch is not None and total_epochs is not None else 1.0
            if torch.rand(1).item() < alpha:
                tgt = targets.view(B, self.max_stages, self.num_teeth * 6)
                tgt = torch.nn.functional.pad(tgt, (0, self.embed_dim - self.num_teeth * 6))
                start_token = torch.zeros(B, 1, self.embed_dim, device=x.device)
                tgt = torch.cat([start_token, tgt[:, :-1, :]], dim=1)
                x = x.unsqueeze(1).repeat(1, self.max_stages, 1, 1).view(B, -1, self.embed_dim) + tgt
                for stage in self.stages:
                    x, feat_size = stage(x, feat_size)
                features = self.norm(x)  # [B, max_stages * num_teeth, dim]
                features = features.view(B, self.max_stages, self.num_teeth, -1)  # [B, max_stages, num_teeth, dim]
                x = self.out_layer(features)  # [B, max_stages, num_teeth, 6]
            else:
                features_list = []
                outputs = []
                tgt = torch.zeros(B, 1, self.embed_dim, device=x.device)
                for t in range(self.max_stages):
                    tgt_t = tgt + self.pos_embed[:, :1, :]
                    x_t = x
                    for stage in self.stages:
                        x_t, feat_size = stage(x_t, feat_size)
                    x_t = self.norm(x_t)
                    stage_features = x_t.view(B, self.num_teeth, -1)  # [B, num_teeth, dim]
                    stage_out = self.out_layer(stage_features)  # [B, num_teeth, 6]
                    features_list.append(stage_features)
                    outputs.append(stage_out)
                    next_tgt = torch.nn.functional.pad(stage_out, (0, self.embed_dim - self.num_teeth * 6))
                    tgt = torch.cat([tgt, next_tgt], dim=1)
                features = torch.stack(features_list, dim=1)  # [B, max_stages, num_teeth, dim]
                x = torch.stack(outputs, dim=1)  # [B, max_stages, num_teeth, 6]
        else:
            features_list = []
            outputs = []
            for t in range(self.max_stages):
                x_t = x
                for stage in self.stages:
                    x_t, feat_size = stage(x_t, feat_size)
                x_t = self.norm(x_t)
                stage_features = x_t.view(B, self.num_teeth, -1)  # [B, num_teeth, dim]
                stage_out = self.out_layer(stage_features)  # [B, num_teeth, 6]
                features_list.append(stage_features)
                outputs.append(stage_out)
            features = torch.stack(features_list, dim=1)  # [B, max_stages, num_teeth, dim]
            x = torch.stack(outputs, dim=1)  # [B, max_stages, num_teeth, 6]

        logger.debug(f"MViTv2 output range: min={x.min().item():.4f}, max={x.max().item():.4f}")
        return features, x