import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional_encoding = nn.Parameter(torch.zeros(14, d_model))  # Learnable for 14 teeth

    def forward(self, x):
        # x: (batch_size, 14, 256)
        x = x + self.positional_encoding  # Add positional encoding
        out = self.transformer_encoder(x)  # (batch_size, 14, 256)
        return out

class PredictionHeads(nn.Module):
    def __init__(self, d_model=256):
        super(PredictionHeads, self).__init__()
        # Translation magnitude head
        self.trans_magnitude = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # |Left/Right|, |Forward/Backward|, |Extrude/Intrude|
            nn.ReLU()  # Ensure non-negative
        )
        # Rotation magnitude head
        self.rot_magnitude = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # |Buccal/Lingual|, |Mesial/Distal|, |Rotation|
            nn.ReLU()  # Ensure non-negative
        )
        # Direction head (single for all 6 params; can split into trans/rot if needed)
        self.direction = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6),  # Binary direction for all 6 params
            nn.Sigmoid()  # Probabilities (0=negative, 1=positive)
        )
        # Activity head (single for all 6 params; can split into trans/rot if needed)
        self.activity = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6),  # Binary activity for all 6 params
            nn.Sigmoid()  # Probabilities (0=inactive, 1=active)
        )

    def forward(self, x):
        # x: (batch_size, 14, 256)
        trans_mag = self.trans_magnitude(x)  # (batch_size, 14, 3)
        rot_mag = self.rot_magnitude(x)  # (batch_size, 14, 3)
        directions = self.direction(x)  # (batch_size, 14, 6)
        activities = self.activity(x)  # (batch_size, 14, 6)
        return trans_mag, rot_mag, directions, activities

class TransformerModel(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.transformer = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.prediction_heads = PredictionHeads(d_model=d_model)

    def forward(self, x):
        # x: (batch_size, 14, 256) from PointNet
        transformer_out = self.transformer(x)  # (batch_size, 14, 256)
        trans_mag, rot_mag, directions, activities = self.prediction_heads(transformer_out)
        # Combine magnitudes and directions without activity mask
        return trans_mag, rot_mag, directions, activities