import torch
from torch import nn
from einops import rearrange
from timm.models.vision_transformer import Block
class LocalEncoder(nn.Module):
    """
    Processes all 14 teeth using the Mesh_encoder.
    - Configurable embedding dimension and number of teeth.
    """
    def __init__(self, mesh_encoder, embed_dim=768, num_teeth=14):
        super(LocalEncoder, self).__init__()
        self.encoder = mesh_encoder
        self.embed_dim = embed_dim
        self.num_teeth = num_teeth
    
    def forward(self, faces, feats, centers, Fs, cordinates):
        batch_size = feats.shape[0]
        local_features = torch.zeros(batch_size, self.num_teeth, self.embed_dim, device=faces.device)
        
        for tooth_idx in range(self.num_teeth):
            local_features[:, tooth_idx] = self.encoder(
                faces[:, tooth_idx], feats[:, tooth_idx], 
                centers[:, tooth_idx], Fs[:, tooth_idx], 
                cordinates[:, tooth_idx]
            )
        
        return local_features