import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=2, dim_feedforward=512, dropout=0.2):
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
        # Sinusoidal positional encoding with learnable offset
        self.positional_encoding = self._get_sinusoidal_encoding(14, d_model)
        self.pos_offset = nn.Parameter(torch.zeros(14, d_model))
        self.norm = nn.LayerNorm(d_model)

    def _get_sinusoidal_encoding(self, seq_len, d_model):
        position = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        # x: (batch_size, 14, 256)
        pe = self.positional_encoding.to(x.device) + self.pos_offset
        x = x + pe  # Add positional encoding
        x = self.transformer_encoder(x)  # (batch_size, 14, 256)
        x = self.norm(x)  # Additional normalization
        return x

class PredictionHeads(nn.Module):
    def __init__(self, d_model=256):
        super(PredictionHeads, self).__init__()
        # Translation magnitude head
        self.trans_magnitude = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # |Left/Right|, |Forward/Backward|, |Extrude/Intrude|
            nn.ReLU()  # Ensure non-negative
        )
        # Rotation magnitude head
        self.rot_magnitude = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # |Buccal/Lingual|, |Mesial/Distal|, |Rotation|
            nn.ReLU()  # Ensure non-negative
        )
        # Direction head (takes concatenated input: 256*2=512)
        self.direction = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6),  # Binary direction for all 6 params
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_trans, x_rot):
        x_trans = self.norm(x_trans)  # Normalize translation input
        x_rot = self.norm(x_rot)  # Normalize rotation input
        trans_mag = self.trans_magnitude(x_trans)  # (batch_size, 14, 3)
        rot_mag = self.rot_magnitude(x_rot)  # (batch_size, 14, 3)
        # Concatenate transformer outputs for direction head
        x_concat = torch.cat([x_trans, x_rot], dim=-1)  # (batch_size, 14, 512)
        directions = self.direction(x_concat)  # (batch_size, 14, 6)
        return trans_mag, rot_mag, directions

class TransformerModel(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.2):
        super(TransformerModel, self).__init__()
        # Separate transformers for translation and rotation
        self.trans_transformer = TransformerEncoder(d_model, nhead, num_layers//2, dim_feedforward, dropout)
        self.rot_transformer = TransformerEncoder(d_model, nhead, num_layers//2, dim_feedforward, dropout)
        self.prediction_heads = PredictionHeads(d_model=d_model)

    def forward(self, x):
        # x: (batch_size, 14, 256) from PointNet
        trans_out = self.trans_transformer(x)  # (batch_size, 14, 256)
        rot_out = self.rot_transformer(x)  # (batch_size, 14, 256)
        # Pass to prediction heads
        trans_mag, rot_mag, directions = self.prediction_heads(trans_out, rot_out)
        return trans_mag, rot_mag, directions