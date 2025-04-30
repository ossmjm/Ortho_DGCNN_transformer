import torch
import torch.nn as nn
import logging
from typing import List, Optional
from models.CumulativeTransformationModel import CumulativeTransformationModel
from models.DGCNN import DGCNN
from models.TransformerDecoder import TransformerDecoder

class OrthoDGCNNModel(nn.Module):
    def __init__(
        self,
        max_stages: int = 25,
        num_teeth: int = 14,
        num_points: int = 256,
        channels: int = 13,
        embed_dim: int = 384,
        teacher_forcing: bool = False,
        decoder_layers: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        k: int = 20
    ):
        super().__init__()
        self.cumulative_model = CumulativeTransformationModel(num_teeth=num_teeth, embed_dim=embed_dim)
        self.dgcnn = DGCNN(in_channels=channels, embed_dim=embed_dim, num_teeth=num_teeth, num_points=num_points, k=k)
        self.decoder = TransformerDecoder(embed_dim=embed_dim, num_teeth=num_teeth, max_stages=max_stages, num_layers=decoder_layers, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.max_stages = max_stages
        self.num_teeth = num_teeth
        self.embed_dim = embed_dim
        self.teacher_forcing = teacher_forcing
        
        self.feature_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.activity_head = nn.Linear(embed_dim, 1)
        self.param_activity_head = nn.Linear(embed_dim, 6)
        self.type_head = nn.Linear(embed_dim, 4)
        
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

    def forward(self, coordinates, targets=None, cumulative_targets=None, epoch=None, total_epochs=None):
        logger = logging.getLogger('TrainLogger')
        
        if torch.isnan(coordinates).any():
            logger.error("NaN values detected in input coordinates")
            coordinates = torch.nan_to_num(coordinates, nan=0.0, posinf=1.0, neginf=-1.0)
            
        logger.debug(f"size before dgcnn:{coordinates.shape}")
        features = self.dgcnn(coordinates)
        logger.debug(f"size after dgcnn:{features.shape}")
        
        if torch.isnan(features).any():
            logger.error("NaN values detected in DGCNN features")
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
        features = self.feature_norm(features)
        features = torch.clamp(features, -100, 100)
        
        cumulative_features = features.detach()
        cumulative_transforms, cumulative_activity_logits, cumulative_param_activity_logits = self.cumulative_model(cumulative_features)
        
        if torch.isnan(cumulative_transforms).any():
            logger.error("NaN values detected in cumulative_transforms")
            cumulative_transforms = torch.nan_to_num(cumulative_transforms, nan=0.0, posinf=1.0, neginf=-1.0)
            cumulative_transforms = torch.clamp(cumulative_transforms, -100.0, 100.0)
            
        alpha = max(0.0, 1.0 - (epoch / (total_epochs * 0.5))) if epoch is not None and total_epochs is not None else 0.0
        use_cumulative_teacher_forcing = self.training and self.teacher_forcing and cumulative_targets is not None and torch.rand(1).item() < alpha
        cumulative_input = cumulative_targets if use_cumulative_teacher_forcing else cumulative_transforms
        
        if use_cumulative_teacher_forcing and torch.isnan(cumulative_targets).any():
            logger.error("NaN values detected in cumulative_targets")
            cumulative_input = torch.nan_to_num(cumulative_targets, nan=0.0, posinf=1.0, neginf=-1.0)
            cumulative_input = torch.clamp(cumulative_input, -100.0, 100.0)
        
        try:
            decoder_features, transforms_sequence = self.decoder(
                memory=features,
                cumulative_transforms=cumulative_input,
                num_stages=None,
                targets=targets,
                use_teacher_forcing=use_cumulative_teacher_forcing,
                training=self.training
            )
            
            if torch.isnan(decoder_features).any():
                logger.error("NaN values detected in decoder_features")
                decoder_features = torch.nan_to_num(decoder_features, nan=0.0, posinf=1.0, neginf=-1.0)
                
            if torch.isnan(transforms_sequence).any():
                logger.error("NaN values detected in transforms_sequence")
                transforms_sequence = torch.nan_to_num(transforms_sequence, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Detach decoder_features to isolate gradients for activity heads
            decoder_features_detached = decoder_features.detach()
            activity_logits = self.activity_head(decoder_features_detached).squeeze(-1)
            param_activity_logits = self.param_activity_head(decoder_features_detached)
            type_logits = self.type_head(decoder_features_detached)
            
            activity_logits = torch.clamp(activity_logits, -10.0, 10.0)
            param_activity_logits = torch.clamp(param_activity_logits, -10.0, 10.0)
            type_logits = torch.clamp(type_logits, -10.0, 10.0)
            cumulative_activity_logits = torch.clamp(cumulative_activity_logits, -10.0, 10.0)
            cumulative_param_activity_logits = torch.clamp(cumulative_param_activity_logits, -10.0, 10.0)
            
        except RuntimeError as e:
            logger.error(f"Runtime error in forward pass: {e}")
            batch_size = features.size(0)
            decoder_features = torch.zeros(batch_size, self.max_stages, self.num_teeth, self.embed_dim, device=features.device)
            transforms_sequence = torch.zeros(batch_size, self.max_stages, self.num_teeth, 6, device=features.device)
            activity_logits = torch.zeros(batch_size, self.max_stages, self.num_teeth, device=features.device)
            param_activity_logits = torch.zeros(batch_size, self.max_stages, self.num_teeth, 6, device=features.device)
            type_logits = torch.zeros(batch_size, self.max_stages, self.num_teeth, 4, device=features.device)
            cumulative_activity_logits = torch.zeros(batch_size, self.num_teeth, device=features.device)
            cumulative_param_activity_logits = torch.zeros(batch_size, self.num_teeth, 6, device=features.device)
            
        logger.debug(f"OrthoDGCNN output shape: transforms_sequence={transforms_sequence.shape}, "
                    f"activity_logits={activity_logits.shape}, param_activity_logits={param_activity_logits.shape}, "
                    f"type_logits={type_logits.shape}, cumulative_transforms={cumulative_transforms.shape}, "
                    f"cumulative_activity_logits={cumulative_activity_logits.shape}, "
                    f"cumulative_param_activity_logits={cumulative_param_activity_logits.shape}")
                     
        outputs = [transforms_sequence, activity_logits, type_logits, param_activity_logits, 
                  cumulative_transforms, cumulative_activity_logits, cumulative_param_activity_logits]
        
        for i, output in enumerate(outputs):
            if torch.isnan(output).any():
                logger.error(f"NaN values detected in output {i}")
                outputs[i] = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
                
        return outputs