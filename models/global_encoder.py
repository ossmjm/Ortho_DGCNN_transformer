import torch
from torch import nn
from einops import rearrange
from timm.models.vision_transformer import Block
class GlobalEncoder(nn.Module):
    """
    Encodes jaw-wide context using a PointNet-like architecture.
    - Adjustable global embedding dimension.
    """
    def __init__(self, global_embed_dim=512):
        super(GlobalEncoder, self).__init__()
        self.global_embed_dim = global_embed_dim
        self.encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, global_embed_dim)
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
    
    def forward(self, all_vertices):
        if isinstance(all_vertices, list):
            all_vertices = torch.cat([torch.tensor(v, dtype=torch.float32) for v in all_vertices], dim=0)
        all_vertices = all_vertices.to(next(self.parameters()).device)
        global_features = self.encoder(all_vertices)
        global_features = global_features.transpose(0, 1).unsqueeze(0)
        global_features = self.pool(global_features).squeeze(-1)
        return global_features