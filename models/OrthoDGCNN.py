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
        self.type_layers = nn.Sequential(
            nn.Linear(self.in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_teeth * 4)
        )
        self.param_activity_layers = nn.Sequential(
            nn.Linear(self.in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_teeth * 6)
        )
        self.trans_scale = nn.Parameter(torch.tensor(15.0))
        self.rot_scale = nn.Parameter(torch.tensor(45.0))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Parameter):
                nn.init.constant_(m, 1.0)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size * self.max_stages, self.in_dim)
        transforms = self.transform_layers(x)
        transforms = transforms.view(batch_size, self.max_stages, self.num_teeth, 6)
        transforms[:, :, :, :3] *= self.trans_scale.abs()
        transforms[:, :, :, 3:] *= self.rot_scale.abs()
        activity_logits = self.activity_layers(x)
        activity_logits = activity_logits.view(batch_size, self.max_stages, self.num_teeth)
        type_logits = self.type_layers(x)
        type_logits = type_logits.view(batch_size, self.max_stages, self.num_teeth, 4)
        param_activity_logits = self.param_activity_layers(x)
        param_activity_logits = param_activity_logits.view(batch_size, self.max_stages, self.num_teeth, 6)
        
        activity_probs = torch.sigmoid(activity_logits)
        active_mask = (activity_probs > 0.5).float().unsqueeze(-1)
        param_activity_probs = torch.sigmoid(param_activity_logits)
        param_active_mask = (param_activity_probs > 0.5).float()
        transforms = transforms * active_mask * param_active_mask
        
        logger = logging.getLogger('TrainLogger')
        logger.debug(f"TransformHead: transforms min={transforms.min().item():.4f}, max={transforms.max().item():.4f}")
        logger.debug(f"TransformHead: activity_logits min={activity_logits.min().item():.4f}, max={activity_logits.max().item():.4f}")
        logger.debug(f"TransformHead: type_logits min={type_logits.min().item():.4f}, max={type_logits.max().item():.4f}")
        logger.debug(f"TransformHead: param_activity_logits min={param_activity_logits.min().item():.4f}, max={param_activity_logits.max().item():.4f}")
        
        return transforms, activity_logits, type_logits, param_activity_logits

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
        transforms_sequence, activity_logits, type_logits, param_activity_logits = self.transform_head(transformer_out)
        return transforms_sequence, activity_logits, type_logits, param_activity_logits