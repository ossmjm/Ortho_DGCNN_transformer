import torch
import torch.nn as nn
import logging
from models.GRU_cumulativedecoder import GRUToothDecoder
from models.DGCNN import DGCNN
from models.pointnet2 import PointNetPlusPlus

class OrthoDGCNNModel(nn.Module):
    def __init__(
        self,
        num_teeth: int = 14,
        num_points: int = 256,
        channels: int = 4,
        embed_dim: int = 256,
        k: int = 20,
        encoder_type: str = 'dgcnn',
        num_layers: int = 1,
        nhead: int = 8
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.num_teeth = num_teeth
        self.num_layers = num_layers
        self.nhead = nhead
        valid_encoder_types = ['dgcnn', 'pointnet2']
        if encoder_type not in valid_encoder_types:
            raise ValueError(f"Invalid encoder_type: {encoder_type}. Must be one of {valid_encoder_types}")

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
        self.cumulative_model = GRUToothDecoder(
            d_model=embed_dim,
            num_layers=num_layers,
            dim_feedforward=512,
            dropout=0.1
        )
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

    def forward(self, coordinates, cumulative_transforms=None, active_labels=None, direction_labels=None, training=False, epoch=0, total_epochs=100, val_loss=None, is_freeze='none'):
        logger = logging.getLogger('TrainLogger')
        
        if torch.isnan(coordinates).any():
            logger.error("NaN values detected in input coordinates")
            coordinates = torch.nan_to_num(coordinates, nan=0.0, posinf=1.0, neginf=-1.0)
            
        features = self.encoder(coordinates)  # (batch_size, 14, embed_dim)
        logger.debug(f"size after {self.encoder_type}: {features.shape}")
        
        if torch.isnan(features).any():
            logger.error(f"NaN values detected in {self.encoder_type} features")
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
        # Order teeth by x-coordinate for dental arch sequence
        centroids = coordinates.mean(dim=2)  # (batch_size, 14, 3)
        tooth_order = torch.argsort(centroids[:, :, 0], dim=1)  # (batch_size, 14)
        features = features.gather(1, tooth_order.unsqueeze(-1).expand(-1, -1, features.size(-1)))  # (batch_size, 14, embed_dim)
        
        features = self.feature_norm(features)  # (batch_size, 14, embed_dim)
        
        outputs = self.cumulative_model(
            features,
            cumulative_transforms=cumulative_transforms,
            active_labels=active_labels,
            direction_labels=direction_labels,
            training=training,
            epoch=epoch,
            total_epochs=total_epochs,
            val_loss=val_loss,
            is_freeze=is_freeze
        )
        
        # Reorder outputs to match original tooth indices
        inverse_order = torch.argsort(tooth_order, dim=1)  # (batch_size, 14)
        outputs_list = list(outputs)  # Convert tuple to list for modification
        for i in range(len(outputs_list)):
            outputs_list[i] = outputs_list[i].gather(1, inverse_order.unsqueeze(-1).expand(-1, -1, outputs_list[i].size(-1)))
        outputs = tuple(outputs_list)  # Convert back to tuple
        
        for i, output in enumerate(outputs):
            if torch.isnan(output).any():
                logger.error(f"NaN values detected in output {i}")
                outputs_list = list(outputs)  # Convert to list again for modification
                outputs_list[i] = torch.nan_to_num(outputs_list[i], nan=0.0, posinf=1.0, neginf=-1.0)
                outputs = tuple(outputs_list)  # Convert back to tuple
                
        return outputs