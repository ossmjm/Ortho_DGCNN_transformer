import torch
from torch import nn
from einops import rearrange
from timm.models.vision_transformer import Block

class Mesh_encoder(nn.Module):
    """
    Encodes individual tooth features using a vision transformer backbone.
    - Configurable depth, heads, and embedding dimension.
    """
    def __init__(self, channels=13, num_heads=12, encoder_depth=12, embed_dim=768, patch_size=64, norm_layer=nn.LayerNorm):
        super(Mesh_encoder, self).__init__()
        patch_dim = channels
        self.num_patches = 256
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c h p -> b h (p c)', p=patch_size),
            nn.Linear(patch_dim * patch_size, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.dim = embed_dim
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(encoder_depth)])
        self.norm = norm_layer(embed_dim)
        self.pos_embedding = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )
        self.max_pooling2 = nn.MaxPool2d((256, 1))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.normal_(self.cls_token, std=.02)

    def forward(self, faces, feats, centers, Fs, cordinates):
        feats_patches = feats
        centers_patches = centers
        center_of_patches = torch.sum(centers_patches, dim=2) / 64
        pos_emb = self.pos_embedding(center_of_patches)
        batch = feats_patches.shape[0]
        tokens = self.to_patch_embedding(feats_patches)
        tokens = tokens + pos_emb
        for blk in self.blocks:
            tokens = blk(tokens)
        x = self.norm(tokens)
        zero_tokens = torch.zeros((batch, 256 - self.num_patches, self.dim), dtype=torch.float32).to(faces.device)
        tokens = torch.cat((x, zero_tokens), dim=1)
        tokens = self.max_pooling2(tokens).squeeze(1)
        return tokens