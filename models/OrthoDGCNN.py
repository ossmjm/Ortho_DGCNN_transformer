# models/OrthoDGCNN.py
import torch
import torch.nn as nn

class TransformHead(nn.Module):
    def __init__(self, max_stages, num_teeth):
        super(TransformHead, self).__init__()
        self.max_stages = max_stages
        self.num_teeth = num_teeth
        self.in_dim = num_teeth * 6  # Input dimension per stage: num_teeth * 6 (e.g., 84)
        self.out_dim = num_teeth * 6  # Output dimension per stage: num_teeth * 6 (e.g., 84)
        # A small network to refine the transformer output for each stage
        self.layers = nn.Sequential(
            nn.Linear(self.in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, self.out_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input x: (batch_size, max_stages, num_teeth * 6)
        batch_size = x.size(0)
        x = x.view(batch_size * self.max_stages, self.in_dim)  # (batch_size * max_stages, num_teeth * 6)
        out = self.layers(x)  # (batch_size * max_stages, num_teeth * 6)
        out = out.view(batch_size, self.max_stages, self.num_teeth, 6)  # (batch_size, max_stages, num_teeth, 6)
        return out

class OrthoDGCNNModel(nn.Module):
    def __init__(self, dgcnn, transformer, max_stages, num_teeth, embed_dim):
        super(OrthoDGCNNModel, self).__init__()
        self.dgcnn = dgcnn
        self.transformer = transformer
        self.stage_predictor = nn.Sequential(
            nn.Linear(num_teeth * embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.transform_head = TransformHead(
            max_stages=max_stages,
            num_teeth=num_teeth
        )
        self.max_stages = max_stages
        self.num_teeth = num_teeth
        self._init_weights()

    def _init_weights(self):
        for m in self.stage_predictor.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, cordinates, teacher_forcing=None, true_num_stages=None, epoch=None, total_epochs=None):
        batch_size = cordinates.size(0)
        dgcnn_out = self.dgcnn(cordinates)  # Shape: (batch_size, num_teeth * embed_dim)
        
        # Stage prediction
        num_stages_pred = self.stage_predictor(dgcnn_out)
        num_stages_pred = 1 + (self.max_stages - 1) * num_stages_pred
        num_stages_pred_rounded = torch.round(num_stages_pred).clamp(1, self.max_stages)
        print(f"Epoch {epoch+1 if epoch is not None else 'N/A'}: num_stages_pred = {num_stages_pred.tolist()}")
        print(f"Epoch {epoch+1 if epoch is not None else 'N/A'}: num_stages_pred_rounded = {num_stages_pred_rounded.tolist()}")
        
        # Determine whether to use true_num_stages or num_stages_pred
        if self.training and true_num_stages is not None and epoch is not None and total_epochs is not None:
            if epoch < total_epochs * 0.5:
                num_stages = true_num_stages
            else:
                num_stages = num_stages_pred_rounded
        else:
            num_stages = num_stages_pred_rounded
        
        num_stages = num_stages.long()
        
        # Transformer processing with teacher forcing
        transformer_out = self.transformer(dgcnn_out, num_stages, teacher_forcing)  # Shape: (batch_size, max_stages, num_teeth * 6)
        
        # Pass transformer_out to TransformHead
        transforms_sequence = self.transform_head(transformer_out)  # Shape: (batch_size, max_stages, num_teeth, 6)
        
        return transforms_sequence, num_stages_pred