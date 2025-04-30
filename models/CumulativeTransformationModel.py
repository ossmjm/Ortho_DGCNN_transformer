import torch
import torch.nn as nn
import logging

class CumulativeTransformationModel(nn.Module):
    def __init__(self, num_teeth=14, embed_dim=384):
        super().__init__()
        self.num_teeth = num_teeth
        self.embed_dim = embed_dim
        self.logger = logging.getLogger('TrainLogger')
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512, eps=1e-3),  # Increased eps
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256, eps=1e-3),  # Increased eps
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128, eps=1e-3),  # Increased eps
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 6)
        )
        self.activity_head = nn.Linear(embed_dim, 1)
        self.param_activity_head = nn.Linear(embed_dim, 6)
        
        self._init_weights()
        
        # Freeze BatchNorm statistics during early training
        for module in self.mlp:
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
    
    def forward(self, x):
        batch_size, num_teeth, embed_dim = x.size()
        assert num_teeth == self.num_teeth, f"Expected num_teeth={self.num_teeth}, got {num_teeth}"
        assert embed_dim == self.embed_dim, f"Expected embed_dim={self.embed_dim}, got {embed_dim}"
        
        x = x.view(batch_size * num_teeth, embed_dim)
        self.logger.debug(f"MLP input min: {x.min().item():.4f}, max: {x.max().item():.4f}, has_nan: {torch.isnan(x).any().item()}")
        
        transforms = self.mlp(x)
        transforms = torch.clamp(transforms, -100, 100)  # Clamp outputs
        self.logger.debug(f"MLP output (transforms) min: {transforms.min().item():.4f}, max: {transforms.max().item():.4f}, has_nan: {torch.isnan(transforms).any().item()}")
        
        transforms = transforms.view(batch_size, num_teeth, 6)
        activity_logits = self.activity_head(x).view(batch_size, num_teeth)
        param_activity_logits = self.param_activity_head(x).view(batch_size, num_teeth, 6)
        
        self.logger.debug(f"CumulativeTransformationModel output shape: transforms={transforms.shape}, "
                         f"activity_logits={activity_logits.shape}, param_activity_logits={param_activity_logits.shape}")
        return transforms, activity_logits, param_activity_logits