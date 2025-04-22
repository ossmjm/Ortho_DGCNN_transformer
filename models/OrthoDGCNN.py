import torch
import torch.nn as nn
import logging
from typing import List, Optional

class OrthoDGCNNModel(nn.Module):
    def __init__(
        self,
        dgcnn,
        mvit,
        max_stages: int = 25,
        num_teeth: int = 14,
        embed_dim: int = 256,
        teacher_forcing: bool = False,
        depths: List[int] = [1, 2, 11, 2],
        num_heads: List[int] = [4, 4, 8, 8],
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.2
    ):
        super(OrthoDGCNNModel, self).__init__()
        self.dgcnn = dgcnn
        self.mvit = mvit
        self.max_stages = max_stages
        self.num_teeth = num_teeth
        self.embed_dim = embed_dim
        self.teacher_forcing = teacher_forcing
        
        # Classification heads for activity prediction
        # Final stage dim is embed_dim * 4 (e.g., 1024 if embed_dim=256)
        self.activity_head = nn.Linear(embed_dim * 4, 1)  # Predicts per-tooth activity
        self.param_activity_head = nn.Linear(embed_dim * 4, 6)  # Predicts per-parameter activity
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, cordinates, targets=None, epoch=None, total_epochs=None):
        logger = logging.getLogger('TrainLogger')
        
        # DGCNN processes point clouds
        dgcnn_out = self.dgcnn(cordinates)  # [batch_size, 2, 7, embed_dim]
        
        # Pass to MViTv2
        features, transforms_sequence = self.mvit(
            dgcnn_out,
            targets=targets,
            epoch=epoch,
            total_epochs=total_epochs
        )  # features: [batch_size, max_stages, num_teeth, embed_dim * 4]
           # transforms_sequence: [batch_size, max_stages, num_teeth, 6]
        
        # Predict activity logits
        activity_logits = self.activity_head(features).squeeze(-1)  # [batch_size, max_stages, num_teeth]
        
        # Predict parameter activity logits
        param_activity_logits = self.param_activity_head(features)  # [batch_size, max_stages, num_teeth, 6]
        
        logger.debug(f"OrthoDGCNN output shape: {transforms_sequence.shape}")
        return transforms_sequence, activity_logits, param_activity_logits