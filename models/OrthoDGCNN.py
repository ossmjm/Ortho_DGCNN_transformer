import torch
import torch.nn as nn
import logging
from models.GNNmodel import GNNModel
from models.DGCNN import DGCNN
from models.pointnet2 import PointNetPlusPlus
from torch_geometric.utils import dense_to_sparse

class OrthoDGCNNModel(nn.Module):
    def __init__(
        self,
        num_teeth: int = 14,
        num_points: int = 256,
        channels: int = 4,
        embed_dim: int = 256,
        k: int = 20,
        encoder_type: str = 'dgcnn',
        num_layers: int = 2,
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
        self.cumulative_model = GNNModel(
            d_model=embed_dim,
            nhead=nhead,
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

    def _create_fully_connected_edge_index(self, num_nodes):
        """Create edge_index for a fully connected graph with num_nodes nodes."""
        adj = torch.ones(num_nodes, num_nodes, dtype=torch.float)
        adj = adj - torch.eye(num_nodes)  # Remove self-loops
        edge_index, _ = dense_to_sparse(adj)
        return edge_index

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
        
        # Create edge_index for fully connected graph
        edge_index = self._create_fully_connected_edge_index(self.num_teeth)
        edge_index = edge_index.to(coordinates.device)
        
        trans_mag, rot_mag, directions = self.cumulative_model(features, edge_index)
        
        outputs = [trans_mag, rot_mag, directions]
        
        for i, output in enumerate(outputs):
            if torch.isnan(output).any():
                logger.error(f"NaN values detected in output {i}")
                outputs[i] = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
                
        return outputs