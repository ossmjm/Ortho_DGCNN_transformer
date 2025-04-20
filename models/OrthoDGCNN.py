import torch
import torch.nn as nn
import logging

class TransformHead(nn.Module):
    def __init__(self, max_stages, num_teeth):
        super(TransformHead, self).__init__()
        self.max_stages = max_stages
        self.num_teeth = num_teeth
        self.in_dim = num_teeth * 6
        self.out_dim = num_teeth * 6
        self.transform_layers = nn.Sequential(
            nn.Linear(self.in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.out_dim)
        )
        self.activity_layers = nn.Sequential(
            nn.Linear(self.in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_teeth)
        )
        self.trans_scale = 2.0
        self.rot_scale = 30.0
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size * self.max_stages, self.in_dim)
        transforms = self.transform_layers(x)
        transforms = transforms.view(batch_size, self.max_stages, self.num_teeth, 6)
        transforms[:, :, :, :3] *= self.trans_scale
        transforms[:, :, :, 3:] *= self.rot_scale
        activity_logits = self.activity_layers(x)
        activity_logits = activity_logits.view(batch_size, self.max_stages, self.num_teeth)
        
        logger = logging.getLogger('TrainLogger')
        logger.debug(f"TransformHead: transforms min={transforms.min().item():.4f}, max={transforms.max().item():.4f}")
        logger.debug(f"TransformHead: activity_logits min={activity_logits.min().item():.4f}, max={activity_logits.max().item():.4f}")
        
        return transforms, activity_logits

class OrthoDGCNNModel(nn.Module):
    def __init__(self, dgcnn, transformer, max_stages, num_teeth, embed_dim, teacher_forcing=False):
        super(OrthoDGCNNModel, self).__init__()
        self.dgcnn = dgcnn
        self.transformer = transformer
        self.transform_head = TransformHead(max_stages=max_stages, num_teeth=num_teeth)
        self.max_stages = max_stages
        self.num_teeth = num_teeth
        self.embed_dim = embed_dim
        self.teacher_forcing = teacher_forcing
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, cordinates, targets=None, epoch=None, total_epochs=None):
        dgcnn_out = self.dgcnn(cordinates)
        transformer_out = self.transformer(
            dgcnn_out, 
            targets=targets, 
            teacher_forcing=self.teacher_forcing, 
            epoch=epoch, 
            total_epochs=total_epochs
        )
        transforms_sequence, activity_logits = self.transform_head(transformer_out)
        return transforms_sequence, activity_logits