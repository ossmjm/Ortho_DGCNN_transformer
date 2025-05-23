import torch
import torch.nn as nn
import logging
from models.DGCNN import DGCNN
from models.TransformerDecoder import TransformerDecoder
from models.PerToothTransformer import PerToothTransformerDecoder

class OrthoDGCNNModel(nn.Module):
    def __init__(
        self,
        max_stages: int = 25,
        num_teeth: int = 14,
        num_points: int = 256,
        channels: int = 4,
        embed_dim: int = 384,
        teacher_forcing_prob: float = 0.0,
        decoder_layers: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        k: int = 20,
        decoder_type: str = 'per_tooth',
        per_tooth_layers: int = 4,
        per_tooth_heads: int = 8,
        per_tooth_mlp_ratio: float = 4.0
    ):
        super().__init__()
        self.dgcnn = DGCNN(in_channels=4, embed_dim=embed_dim, num_teeth=num_teeth, num_points=num_points, k=k)
        if decoder_type == 'per_tooth':
            self.decoder = PerToothTransformerDecoder(
                embed_dim=embed_dim,
                num_teeth=num_teeth,
                max_stages=max_stages,
                num_layers=per_tooth_layers,
                num_heads=per_tooth_heads,
                mlp_ratio=per_tooth_mlp_ratio
            )
        else:
            self.decoder = TransformerDecoder(
                embed_dim=embed_dim,
                num_teeth=num_teeth,
                max_stages=max_stages,
                num_layers=decoder_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio
            )
        self.max_stages = max_stages
        self.num_teeth = num_teeth
        self.embed_dim = embed_dim
        self.teacher_forcing_prob = teacher_forcing_prob
        self.num_points = num_points
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

    def forward(self, coordinates, targets=None, cumulative_targets=None, activity_targets=None, param_activity_targets=None, num_stages=None, epoch=None, total_epochs=None, val_loss=None, training=True):
        logger = logging.getLogger('TrainLogger')
        
        expected_shape = (-1, self.num_teeth, self.num_points, 4)
        if coordinates.shape[1:] != torch.Size(expected_shape[1:]):
            logger.error(f"Invalid coordinates shape: got {coordinates.shape}, expected {expected_shape}")
            raise RuntimeError(f"Coordinates shape mismatch: got {coordinates.shape}, expected {expected_shape}")
        
        if torch.isnan(coordinates).any():
            logger.error("NaN values detected in input coordinates")
            coordinates = torch.nan_to_num(coordinates, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if targets is not None:
            expected_targets_shape = (-1, self.max_stages, self.num_teeth, 6)
            if targets.shape[1:] != torch.Size(expected_targets_shape[1:]):
                logger.error(f"Invalid targets shape: got {targets.shape}, expected {expected_targets_shape}")
                raise RuntimeError(f"Targets shape mismatch")
        if activity_targets is not None:
            expected_activity_shape = (-1, self.max_stages, self.num_teeth)
            if activity_targets.shape[1:] != torch.Size(expected_activity_shape[1:]):
                logger.error(f"Invalid activity_targets shape: got {activity_targets.shape}, expected {expected_activity_shape}")
                raise RuntimeError(f"Activity_targets shape mismatch")
        if param_activity_targets is not None:
            expected_param_activity_shape = (-1, self.max_stages, self.num_teeth, 6)
            if param_activity_targets.shape[1:] != torch.Size(expected_param_activity_shape[1:]):
                logger.error(f"Invalid param_activity_targets shape: got {param_activity_targets.shape}, expected {expected_param_activity_shape}")
                raise RuntimeError(f"Param_activity_targets shape mismatch")
        if cumulative_targets is not None:
            expected_cumulative_shape = (-1, self.num_teeth, 6)
            if cumulative_targets.shape[1:] != torch.Size(expected_cumulative_shape[1:]):
                logger.error(f"Invalid cumulative_targets shape: got {cumulative_targets.shape}, expected {expected_cumulative_shape}")
                raise RuntimeError(f"Cumulative_targets shape mismatch")
        
        features = self.dgcnn(coordinates)
        
        expected_features_shape = (-1, self.num_teeth, self.embed_dim)
        if features.shape[1:] != torch.Size(expected_features_shape[1:]):
            logger.error(f"Invalid features shape: got {features.shape}, expected {expected_features_shape}")
            raise RuntimeError(f"Features shape mismatch: got {features.shape}, expected {expected_features_shape}")
        
        if torch.isnan(features).any():
            logger.error("NaN values detected in DGCNN features")
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
        features = self.feature_norm(features)
        
        outputs = self.decoder(
            memory=features,
            cumulative_transforms=cumulative_targets,
            num_stages=num_stages,
            targets=targets,
            activity_targets=activity_targets,
            param_activity_targets=param_activity_targets,
            use_teacher_forcing=self.teacher_forcing_prob if training else 0.0,
            training=training,
            epoch=epoch,
            total_epochs=total_epochs,
            val_loss=val_loss
        )
        
        transforms_sequence, activity_logits, param_activity_logits, stage_activity_logits = outputs
        
        # Extract stage_weights from decoder if it's PerToothTransformerDecoder
        stage_weights = None
        if isinstance(self.decoder, PerToothTransformerDecoder):
            stage_weights = torch.softmax(self.decoder.stage_weights, dim=1)
        
        if torch.isnan(transforms_sequence).any():
            logger.error("NaN values detected in transforms_sequence")
            transforms_sequence = torch.nan_to_num(transforms_sequence, nan=0.0, posinf=1.0, neginf=-1.0)
        
        logger.debug(f"OrthoDGCNN output shape: transforms_sequence={transforms_sequence.shape}, "
                    f"activity_logits={activity_logits.shape}, param_activity_logits={param_activity_logits.shape}, "
                    f"stage_activity_logits={stage_activity_logits.shape}, "
                    f"stage_weights_shape={stage_weights.shape if stage_weights is not None else 'None'}")
                     
        for i, output in enumerate(outputs):
            if torch.isnan(output).any():
                logger.error(f"NaN values detected in output {i}")
                outputs[i] = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
                
        return outputs + [stage_weights]