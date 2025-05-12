import torch
import torch.nn as nn
import logging
from models.CumulativeTransformationModel import CumulativeTransformationModel
from models.DGCNN import DGCNN

class OrthoDGCNNModel(nn.Module):
    def __init__(
        self,
        num_teeth: int = 14,
        num_points: int = 256,
        channels: int = 4,
        embed_dim: int = 384,
        k: int = 20
    ):
        super().__init__()
        self.dgcnn = DGCNN(in_channels=channels, embed_dim=embed_dim, num_teeth=num_teeth, num_points=num_points, k=k)
        self.cumulative_model = CumulativeTransformationModel(num_teeth=num_teeth, embed_dim=embed_dim)
        self.num_teeth = num_teeth
        self.embed_dim = embed_dim
        
        self.feature_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, coordinates):
        logger = logging.getLogger('TrainLogger')
        
        if torch.isnan(coordinates).any():
            logger.error("NaN values detected in input coordinates")
            coordinates = torch.nan_to_num(coordinates, nan=0.0, posinf=1.0, neginf=-1.0)
            
        features = self.dgcnn(coordinates)
        logger.debug(f"size after dgcnn: {features.shape}")
        
        if torch.isnan(features).any():
            logger.error("NaN values detected in DGCNN features")
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
        features = self.feature_norm(features)
        
        cumulative_transforms, cumulative_activity_logits, cumulative_param_activity_logits = self.cumulative_model(features)
        
        if torch.isnan(cumulative_transforms).any():
            logger.info("NaN values detected in cumulative_transforms")
            cumulative_transforms = torch.nan_to_num(cumulative_transforms, nan=0.0, posinf=1.0, neginf=-1.0)
        # print(cumulative_transforms)   
        outputs = [cumulative_transforms, cumulative_activity_logits, cumulative_param_activity_logits]
        
        for i, output in enumerate(outputs):
            if torch.isnan(output).any():
                print(f"NaN values detected in output {i}")
                outputs[i] = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
                
        return outputs