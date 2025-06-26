import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import logging

class GNNModel(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=2, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout

        # Layer normalization for input features
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

        # Translation GAT branch
        self.trans_gat = nn.ModuleList([
            GATConv(d_model, d_model // nhead, heads=nhead, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.trans_norm = nn.ModuleList([
            nn.LayerNorm(d_model, eps=1e-6) for _ in range(num_layers)
        ])
        self.trans_mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 3)
        )

        # Rotation GAT branch
        self.rot_gat = nn.ModuleList([
            GATConv(d_model, d_model // nhead, heads=nhead, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.rot_norm = nn.ModuleList([
            nn.LayerNorm(d_model, eps=1e-6) for _ in range(num_layers)
        ])
        self.rot_mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 3)
        )

        # Direction GAT branch
        self.direction_gat = GATConv(d_model * 2, d_model // nhead, heads=nhead, dropout=dropout)
        self.direction_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.direction = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 6),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, GATConv):
                nn.init.xavier_uniform_(m.lin.weight)  # Updated to use lin instead of lin_src
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index):
        logger = logging.getLogger('TrainLogger')
        
        if torch.isnan(x).any():
            logger.error("NaN values detected in input features")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        # Normalize input features
        batch_size, num_teeth, d_model = x.shape
        x = self.norm(x)  # (batch_size, 14, d_model)

        # Translation branch
        trans_features = x.view(-1, d_model)  # (batch_size * 14, d_model)
        for gat, norm in zip(self.trans_gat, self.trans_norm):
            trans_features = gat(trans_features, edge_index)
            trans_features = F.relu(trans_features)
            trans_features = norm(trans_features)
            trans_features = F.dropout(trans_features, p=self.dropout, training=self.training)
        trans_mag = self.trans_mlp(trans_features).view(batch_size, num_teeth, 3).abs()

        # Rotation branch
        rot_features = x.view(-1, d_model)  # (batch_size * 14, d_model)
        for gat, norm in zip(self.rot_gat, self.rot_norm):
            rot_features = gat(rot_features, edge_index)
            rot_features = F.relu(rot_features)
            rot_features = norm(rot_features)
            rot_features = F.dropout(rot_features, p=self.dropout, training=self.training)
        rot_mag = self.rot_mlp(rot_features).view(batch_size, num_teeth, 3).abs()

        # Direction branch
        concat_features = torch.cat([trans_features, rot_features], dim=-1)  # (batch_size * 14, d_model * 2)
        direction_features = self.direction_gat(concat_features, edge_index)
        direction_features = F.relu(direction_features)
        direction_features = self.direction_norm(direction_features)
        direction_features = F.dropout(direction_features, p=self.dropout, training=self.training)
        directions = self.direction(direction_features).view(batch_size, num_teeth, 6)

        # Check for NaN
        for output, name in [(trans_mag, 'trans_mag'), (rot_mag, 'rot_mag'), (directions, 'directions')]:
            if torch.isnan(output).any():
                logger.error(f"NaN values detected in {name}")
                output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)

        return trans_mag, rot_mag, directions