import torch
import torch.nn as nn
import logging
from models.DGCNN import DGCNN
from models.pointnet2 import PointNetPlusPlus
from models.TransformerDecoder import TransformerDecoder
from models.PerToothTransformer import PerToothTransformerDecoder

class OrthoDGCNNModel(nn.Module):
    def __init__(
        self,
        max_stages: int = 25,
        num_teeth: int = 14,
        num_points: int = 256,
        channels: int = 3,
        embed_dim: int = 384,
        decoder_layers: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        k: int = 20,
        decoder_type: str = 'transformer',
        encoder_type: str = 'dgcnn'  # New parameter to choose encoder
    ):
        super().__init__()
        self.encoder_type = encoder_type
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

        # Validate decoder type
        valid_decoder_types = ['transformer', 'per_tooth']
        if decoder_type not in valid_decoder_types:
            raise ValueError(f"Invalid decoder_type: {decoder_type}. Must be one of {valid_decoder_types}")

        if decoder_type == 'transformer':
            self.decoder = TransformerDecoder(
                embed_dim=embed_dim,
                num_teeth=num_teeth,
                max_stages=max_stages,
                num_layers=decoder_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio
            )
        elif decoder_type == 'per_tooth':
            self.decoder = PerToothTransformerDecoder(
                embed_dim=embed_dim,
                num_teeth=num_teeth,
                max_stages=max_stages,
                num_heads=num_heads,
                num_layers=decoder_layers,
                mlp_ratio=mlp_ratio
            )

        self.max_stages = max_stages
        self.num_teeth = num_teeth
        self.embed_dim = embed_dim
        self.num_points = num_points
        self.decoder_type = decoder_type
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

    def forward(self, coordinates, cumulative_targets, num_stages=None, targets=None, directions=None, training=True, epoch=0, total_epochs=100, val_loss=None):
        logger = logging.getLogger('TrainLogger')

        expected_shape = (-1, self.num_teeth, self.num_points, 3)
        if coordinates.shape[1:] != torch.Size(expected_shape[1:]):
            logger.error(f"Invalid coordinates shape: thro {coordinates.shape}, expected {expected_shape}")
            raise RuntimeError(f"Coordinates shape mismatch: got {coordinates.shape}, expected {expected_shape}")

        if torch.isnan(coordinates).any():
            logger.error("NaN values detected in input coordinates")
            coordinates = torch.nan_to_num(coordinates, nan=0.0, posinf=1.0, neginf=-1.0)

        if cumulative_targets is not None:
            expected_cumulative_shape = (-1, self.num_teeth, 6)
            if cumulative_targets.shape[1:] != torch.Size(expected_cumulative_shape[1:]):
                logger.error(f"Invalid cumulative_targets shape: got {cumulative_targets.shape}, expected {expected_cumulative_shape}")
                raise RuntimeError(f"Cumulative_targets shape mismatch")

        if training:
            if targets is not None:
                expected_targets_shape = (-1, self.max_stages, self.num_teeth, 6)
                if targets.shape[1:] != torch.Size(expected_targets_shape[1:]):
                    logger.error(f"Invalid targets shape: got {targets.shape}, expected {expected_targets_shape}")
                    raise RuntimeError(f"Targets shape mismatch")
            if directions is not None:
                expected_directions_shape = (-1, self.max_stages, self.num_teeth, 6)
                if directions.shape[1:] != torch.Size(expected_directions_shape[1:]):
                    logger.error(f"Invalid directions shape: got {directions.shape}, expected {expected_directions_shape}")
                    raise RuntimeError(f"Directions shape mismatch")

        features = self.encoder(coordinates)

        expected_features_shape = (-1, self.num_teeth, self.embed_dim)
        if features.shape[1:] != torch.Size(expected_features_shape[1:]):
            logger.error(f"Invalid features shape: got {features.shape}, expected {expected_features_shape}")
            raise RuntimeError(f"Features shape mismatch: got {features.shape}, expected {expected_features_shape}")

        if torch.isnan(features).any():
            logger.error(f"NaN values detected in {self.encoder_type} features")
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        features = self.feature_norm(features)

        if not training:
            num_stages = None

        logger.debug(f"Using encoder type: {self.encoder_type}, decoder type: {self.decoder_type}")

        outputs = self.decoder(
            memory=features,
            cumulative_transforms=cumulative_targets,
            num_stages=num_stages,
            targets=targets,
            directions=directions,
            training=training,
            epoch=epoch,
            total_epochs=total_epochs,
            val_loss=val_loss
        )
        ratios_sequence, directions_sequence = outputs

        if torch.isnan(ratios_sequence).any():
            logger.error("NaN values detected in ratios_sequence")
            ratios_sequence = torch.nan_to_num(ratios_sequence, nan=0.0, posinf=1.0, neginf=-1.0)

        if torch.isnan(directions_sequence).any():
            logger.error("NaN values detected in directions_sequence")
            directions_sequence = torch.nan_to_num(directions_sequence, nan=0.0, posinf=1.0, neginf=-1.0)

        logger.debug(f"OrthoPointNet2DGCNN output shape: ratios_sequence={ratios_sequence.shape}, "
                     f"directions_sequence={directions_sequence.shape}")

        return [ratios_sequence, directions_sequence]