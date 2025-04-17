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
            nn.Linear(128, max_stages)  # Output logits for max_stages classes (1 to max_stages)
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
        dgcnn_out = self.dgcnn(cordinates)
        
        # Stage prediction (classification)
        stage_logits = self.stage_predictor(dgcnn_out)  # Shape: (batch_size, max_stages)
        # Convert logits to predicted stages (1 to max_stages)
        num_stages_pred = torch.argmax(stage_logits, dim=1) + 1  # Shape: (batch_size,), values in [1, max_stages]
        
        # print(f"Epoch {epoch+1 if epoch is not None else 'N/A'}: stage_logits = {stage_logits.tolist()}")
        print(f"Epoch {epoch+1 if epoch is not None else 'N/A'}: num_stages_pred = {num_stages_pred.tolist()}")
        
        # Determine whether to use true_num_stages or num_stages_pred with gradual transition
        if self.training and true_num_stages is not None and epoch is not None and total_epochs is not None:
            alpha = min(1.0, epoch / (total_epochs * 0.75))  # Extend transition to 75% of epochs
            num_stages = alpha * num_stages_pred + (1 - alpha) * true_num_stages
            num_stages = torch.round(num_stages).clamp(1, self.max_stages).long()
        else:
            num_stages = num_stages_pred
        
        # print(f"Epoch {epoch+1 if epoch is not None else 'N/A'}: num_stages (after transition) = {num_stages.tolist()}")
        
        # Transformer processing with teacher forcing
        transformer_out = self.transformer(dgcnn_out, num_stages, teacher_forcing)
        
        # Pass transformer_out to TransformHead
        transforms_sequence = self.transform_head(transformer_out)
        
        return transforms_sequence, stage_logits  # Return logits for cross-entropy loss