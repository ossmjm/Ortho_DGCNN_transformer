# models/OrthoDGCNN.py
import torch
import torch.nn as nn

class TransformHead(nn.Module):
    def __init__(self, max_stages, num_teeth):
        super(TransformHead, self).__init__()
        self.max_stages = max_stages
        self.num_teeth = num_teeth
        self.in_dim = num_teeth * 6
        self.out_dim = num_teeth * 6
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
        batch_size = x.size(0)
        x = x.view(batch_size * self.max_stages, self.in_dim)
        out = self.layers(x)
        out = out.view(batch_size, self.max_stages, self.num_teeth, 6)
        return out

class OrthoDGCNNModel(nn.Module):
    def __init__(self, dgcnn, transformer, max_stages, num_teeth, embed_dim):
        super(OrthoDGCNNModel, self).__init__()
        self.dgcnn = dgcnn
        self.transformer = transformer
        self.stage_predictor = nn.Sequential(
            nn.Linear(num_teeth * embed_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Softplus()
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
                    if m.bias.shape[0] == 1:
                        nn.init.constant_(m.bias, 1.0)
                    else:
                        nn.init.constant_(m.bias, 0)

    def forward(self, cordinates, teacher_forcing=None, true_num_stages=None, epoch=None, total_epochs=None):
        batch_size = cordinates.size(0)
        dgcnn_out = self.dgcnn(cordinates)
        
        # Stage prediction
        num_stages_pred = self.stage_predictor(dgcnn_out)
        num_stages_pred = torch.clamp(num_stages_pred, min=1, max=self.max_stages)
        num_stages_pred_rounded = torch.round(num_stages_pred).clamp(1, self.max_stages)
        print(f"Epoch {epoch+1 if epoch is not None else 'N/A'}: num_stages_pred = {num_stages_pred.tolist()}")
        print(f"Epoch {epoch+1 if epoch is not None else 'N/A'}: num_stages_pred_rounded = {num_stages_pred_rounded.tolist()}")
        
        # Determine whether to use true_num_stages or num_stages_pred with gradual transition
        if self.training and true_num_stages is not None and epoch is not None and total_epochs is not None:
            alpha = min(1.0, epoch / (total_epochs * 0.5))
            num_stages = alpha * num_stages_pred_rounded + (1 - alpha) * true_num_stages.unsqueeze(-1)
            num_stages = torch.round(num_stages).clamp(1, self.max_stages).long()
        else:
            num_stages = num_stages_pred_rounded.long()
        
        # Squeeze num_stages to ensure shape (batch_size,)
        num_stages = num_stages.squeeze(-1)
        print(f"Epoch {epoch+1 if epoch is not None else 'N/A'}: num_stages (after squeeze) = {num_stages.tolist()}")
        
        # Transformer processing with teacher forcing
        transformer_out = self.transformer(dgcnn_out, num_stages, teacher_forcing)
        
        # Pass transformer_out to TransformHead
        transforms_sequence = self.transform_head(transformer_out)
        
        return transforms_sequence, num_stages_pred