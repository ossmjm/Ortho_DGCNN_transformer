import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.3):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-3)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.norm(x + residual)
        return x

class CumulativeTransformationModel(nn.Module):
    def __init__(self, num_teeth=14, embed_dim=384, num_heads=4):
        super().__init__()
        self.num_teeth = num_teeth
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.logger = logging.getLogger('TrainLogger')
        
        # Attention layer to capture inter-tooth relationships
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
                
        self.attn_ffn = FeedForward(embed_dim, embed_dim * 2)
        
        # Enhanced MLP for transformation prediction
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.Linear(512, 450),
            nn.BatchNorm1d(450, eps=1e-3),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(450, 256),
            nn.Linear(256, 200),
            nn.BatchNorm1d(200, eps=1e-3),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(200, 128),
            nn.BatchNorm1d(128, eps=1e-3),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.cumulative_transform = nn.Linear(64, 6) # [2.5,1.5,0.3,0.4,0.5]
        # Enhanced activity head
        self.activity_head = nn.Linear(64, 1) #[1] [0]
        
        # Enhanced parameter activity head
        self.param_activity_head = nn.Linear(64, 6)# [0,1,1,0,1,1]
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-3)
        self.final_norm = nn.LayerNorm(6, eps=1e-3)
        
        self._init_weights()
        
        # Freeze BatchNorm statistics during early training
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        batch_size, num_teeth, embed_dim = x.size()
        assert num_teeth == self.num_teeth, f"Expected num_teeth={self.num_teeth}, got {num_teeth}"
        assert embed_dim == self.embed_dim, f"Expected embed_dim={self.embed_dim}, got {embed_dim}"
        
        # Apply attention to capture inter-tooth relationships
        x, _ = self.attention(x, x, x)
        x = self.attn_ffn(x)
        x = self.norm(x)
        
        self.logger.debug(f"Attention output min: {x.min().item():.4f}, max: {x.max().item():.4f}, has_nan: {torch.isnan(x).any().item()}")
        
        # Flatten for MLP processing
        x_flat = x.view(batch_size * num_teeth, embed_dim)
        
        # Predict transformations
        out = self.mlp(x_flat)
        activity_logits = self.activity_head(out).view(batch_size, num_teeth)
        param_activity_logits = self.param_activity_head(out).view(batch_size, num_teeth, 6)
        
        # Apply hierarchical masking
        activity_preds = torch.sigmoid(activity_logits) > 0.5
        param_activity_preds = torch.sigmoid(param_activity_logits) > 0.5
        
        transforms = self.cumulative_transform(out)
        transforms = transforms.view(batch_size, num_teeth, 6)
        transforms = self.final_norm(transforms)
        
        self.logger.debug(f"MLP output (transforms) min: {transforms.min().item():.4f}, max: {transforms.max().item():.4f}, has_nan: {torch.isnan(transforms).any().item()}")
        
        # Apply activity and parameter activity masks
        activity_mask = activity_preds.float().unsqueeze(-1)  # [batch_size, num_teeth, 1]
        param_activity_mask = param_activity_preds.float()    # [batch_size, num_teeth, 6]
        #print(f'activity_mask: {activity_preds},param_mask: {param_activity_preds}')
        transforms = transforms * activity_mask * param_activity_mask
        
        self.logger.debug(f"CumulativeTransformationModel output shape: transforms={transforms.shape}, "
                         f"activity_logits={activity_logits.shape}, param_activity_logits={param_activity_logits.shape}")
        
        return transforms, activity_logits, param_activity_logits
    