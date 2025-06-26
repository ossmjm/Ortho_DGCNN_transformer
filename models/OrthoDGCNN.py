import torch
import torch.nn as nn
import logging
from models.CumulativeTransformationModel import TransformerModel
from models.DGCNN import DGCNN
from models.pointnet2 import PointNetPlusPlus

class OrthoDGCNNModel(nn.Module):
    def __init__(
        self,
        num_teeth: int = 14,
        num_points: int = 256,
        channels: int = 4,
        embed_dim: int = 256,  # Must match TransformerModel's d_model
        k: int = 20,
        encoder_type: str = 'dgcnn',  # Choose encoder: 'dgcnn' or 'pointnet2'
        num_layers: int = 4,
        nhead: int = 8
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.num_layers = num_layers
        self.nhead = nhead
        valid_encoder_types = ['dgcnn', 'pointnet2']
        if encoder_type not in valid_encoder_types:
            raise ValueError(f"Invalid encoder_type: {encoder_type}. Must be one of {valid_encoder_types}")

        # Initialize the appropriate encoder
        if encoder_type == 'dgcnn':
            self.encoder = DGCNN(
                in_channels=channels,
                embed_dim=embed_dim,
                num_teeth=num_teeth,
                num_points=num_points,
                k=k
            )
        elif encoder_type == 'pointnet2':
            self.encoder = PointNetPlusPlus(
                in_channels=channels,
                embed_dim=embed_dim,
                num_teeth=num_teeth,
                num_points=num_points
            )
        self.cumulative_model = TransformerModel(
            d_model=embed_dim,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=512,
            dropout=0.1
        )
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
            
        features = self.encoder(coordinates)  # (batch_size, 14, embed_dim)
        logger.debug(f"size after {self.encoder_type}: {features.shape}")
        
        if torch.isnan(features).any():
            logger.error(f"NaN values detected in {self.encoder_type} features")
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
        features = self.feature_norm(features)  # (batch_size, 14, embed_dim)
        
        trans_mag, rot_mag, directions = self.cumulative_model(features)
        # trans_mag: (batch_size, 14, 3) for |Left/Right|, |Forward/Backward|, |Extrude/Intrude|
        # rot_mag: (batch_size, 14, 3) for |Buccal/Lingual|, |Mesial/Distal|, |Rotation|
        # directions: (batch_size, 14, 6) for direction probabilities (0=negative, 1=positive)
        # activities: (batch_size, 14, 6) for activity probabilities (0=inactive, 1=active)
        
        outputs = [trans_mag, rot_mag, directions]
        
        for i, output in enumerate(outputs):
            if torch.isnan(output).any():
                logger.error(f"NaN values detected in output {i}")
                outputs[i] = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
                
        return outputs