import torch
import torch.nn as nn
import logging
from models.DGCNN import DGCNN
from models.TransformerDecoder import TransformerDecoder
# In OrthoDGCNN_decoder.py
from models.PerToothTransformer import PerToothTransformerDecoder

class OrthoDGCNNModel(nn.Module):
    def __init__(
        self,
        max_stages: int = 25,
        num_teeth: int = 14,
        num_points: int = 256,
        channels: int = 4,
        embed_dim: int = 384,
        teacher_forcing: bool = False,
        decoder_layers: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        k: int = 20,
        decoder_type: str = 'per_tooth',  # New argument
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
        self.teacher_forcing = teacher_forcing
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

    def forward(self, coordinates, targets=None, cumulative_targets=None, num_stages=None, epoch=None, total_epochs=None):
        logger = logging.getLogger('TrainLogger')
        
        # Validate input coordinates shape
        expected_shape = (-1, self.num_teeth, self.num_points,4)  # channels=4
        if coordinates.shape[1:] != torch.Size(expected_shape[1:]):
            logger.error(f"Invalid coordinates shape: got {coordinates.shape}, expected {expected_shape}")
            raise RuntimeError(f"Coordinates shape mismatch: got {coordinates.shape}, expected {expected_shape}")
        
        if torch.isnan(coordinates).any():
            logger.error("NaN values detected in input coordinates")
            coordinates = torch.nan_to_num(coordinates, nan=0.0, posinf=1.0, neginf=-1.0)
            
        features = self.dgcnn(coordinates)
        
        # Validate features shape
        expected_features_shape = (-1, self.num_teeth, self.embed_dim)
        if features.shape[1:] != torch.Size(expected_features_shape[1:]):
            logger.error(f"Invalid features shape: got {features.shape}, expected {expected_features_shape}")
            raise RuntimeError(f"Features shape mismatch: got {features.shape}, expected {expected_features_shape}")
        
        if torch.isnan(features).any():
            logger.error("NaN values detected in DGCNN features")
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
        features = self.feature_norm(features)
        
        alpha = max(0.0, 1.0 - (epoch / (total_epochs * 0.5))) if epoch is not None and total_epochs is not None else 0.0
        use_teacher_forcing = self.training and self.teacher_forcing and targets is not None and torch.rand(1).item() < alpha
        
        # try:
        transforms_sequence, activity_logits, param_activity_logits = self.decoder(
            memory=features,
            cumulative_transforms=cumulative_targets,
            num_stages=num_stages,
            targets=targets,
            use_teacher_forcing=use_teacher_forcing,
            training=self.training
        )
        
        # if torch.isnan(decoder_features).any():
        #     logger.error("NaN values detected in decoder_features")
        #     decoder_features = torch.nan_to_num(decoder_features, nan=0.0, posinf=1.0, neginf=-1.0)
            
        if torch.isnan(transforms_sequence).any():
            logger.error("NaN values detected in transforms_sequence")
            transforms_sequence = torch.nan_to_num(transforms_sequence, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # decoder_features_detached = decoder_features.detach()
        # type_logits = self.type_head(decoder_features_detached)
        
        # except RuntimeError as e:
        #     logger.error(f"Runtime error in forward pass: {e}")
        #     batch_size = features.size(0)
        #     decoder_features = torch.zeros(batch_size, self.max_stages, self.num_teeth, self.embed_dim, device=features.device, requires_grad=True)
        #     transforms_sequence = torch.zeros(batch_size, self.max_stages, self.num_teeth, 6, device=features.device, requires_grad=True)
        #     activity_logits = torch.zeros(batch_size, self.max_stages, self.num_teeth, device=features.device, requires_grad=True)
        #     param_activity_logits = torch.zeros(batch_size, self.max_stages, self.num_teeth, 6, device=features.device, requires_grad=True)
        #     type_logits = torch.zeros(batch_size, self.max_stages, self.num_teeth, 4, device=features.device, requires_grad=True)
            
        logger.debug(f"OrthoDGCNN output shape: transforms_sequence={transforms_sequence.shape}, "
                    f"activity_logits={activity_logits.shape}, param_activity_logits={param_activity_logits.shape}, ")
                     
        outputs = [transforms_sequence, activity_logits, param_activity_logits]
        
        for i, output in enumerate(outputs):
            if torch.isnan(output).any():
                logger.error(f"NaN values detected in output {i}")
                outputs[i] = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
                
        return outputs