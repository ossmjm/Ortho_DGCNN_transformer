import torch
import torch.nn as nn
import logging

class CumulativeTransformationModel(nn.Module):
    def __init__(self, embed_dim=256, num_teeth=14, num_heads=4):
        super(CumulativeTransformationModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_teeth = num_teeth
        self.num_heads = num_heads
        
        # Convolutional layers to process per-tooth features
        self.conv = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim // 2, kernel_size=(2, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),  # Reduce a and b
            nn.BatchNorm3d(embed_dim // 2),
            nn.ReLU(),
            nn.Conv3d(embed_dim // 2, embed_dim // 4, kernel_size=(1, 1, 2), stride=1, padding=0),  # Further reduce
            nn.BatchNorm3d(embed_dim // 4),
            nn.ReLU()
        )
        
        # MLP for transformation prediction
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim // 4, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 6)  # 6 transformation parameters
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)  # Initialize scale to 1
                nn.init.zeros_(m.bias)   # Initialize shift to 0
    
    def forward(self, x):
        logger = logging.getLogger('TrainLogger')
        logger.debug(f"CumulativeTransformationModel input shape={x.shape}")
        
        # Input: [batch_size, 2, 14, 4, embed_dim]
        batch_size = x.size(0)
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # [batch_size, embed_dim, 2, 14, 4]
        
        # Apply convolutions
        x = self.conv(x)  # [batch_size, embed_dim // 4, 1, 14, 1]
        x = x.squeeze(2).squeeze(4)  # [batch_size, embed_dim // 4, 14]
        x = x.permute(0, 2, 1).contiguous()  # [batch_size, 14, embed_dim // 4]
        
        # Predict transformations
        transforms = self.mlp(x)  # [batch_size, 14, 6]
        
        logger.debug(f"CumulativeTransformationModel output shape={transforms.shape}")
        return transforms