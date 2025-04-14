# models/OrthoDGCNN.py
import torch
import torch.nn as nn

class Transform_Head(nn.Module):
    def __init__(self, in_dim, max_stages, num_teeth):
        super(Transform_Head, self).__init__()
        self.max_stages = max_stages
        self.num_teeth = num_teeth
        self.out_dim = max_stages * num_teeth * 6  # e.g., 20 * 14 * 6 = 1680
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.out_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)
        out = self.layers(x)  # (batch_size, max_stages * num_teeth * 6)
        out = out.view(batch_size, self.max_stages, self.num_teeth, 6)  # (batch_size, max_stages, num_teeth, 6)
        return out

class OrthoDGCNNModel(nn.Module):
    def __init__(self, dgcnn, transformer, max_stages, num_teeth, embed_dim):
        super(OrthoDGCNNModel, self).__init__()
        self.dgcnn = dgcnn
        self.transformer = transformer
        self.stage_predictor = nn.Sequential(
            nn.Linear(num_teeth * embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        self.transform_head = Transform_Head(
            in_dim=num_teeth * embed_dim,
            max_stages=max_stages,
            num_teeth=num_teeth
        )
        self.max_stages = max_stages
        self.num_teeth = num_teeth

    def forward(self, cordinates, teacher_forcing=None, true_num_stages=None, epoch=None, total_epochs=None):
        batch_size = cordinates.size(0)
        dgcnn_out = self.dgcnn(cordinates)
        
        # Stage prediction
        num_stages_pred = self.stage_predictor(dgcnn_out)  # Shape: (batch_size, 1)
        print(f"Epoch {epoch+1 if epoch is not None else 'N/A'}: num_stages_pred = {num_stages_pred.tolist()}")
        
        # Determine whether to use true_num_stages or num_stages_pred
        if self.training and true_num_stages is not None and epoch is not None and total_epochs is not None:
            if epoch < total_epochs * 0.5:  # e.g., 50 epochs out of 100
                num_stages = true_num_stages
            else:
                num_stages = num_stages_pred
        else:
            num_stages = num_stages_pred
        
        num_stages = num_stages.long()  # Convert to integer type for indexing
        
        # Transformer processing with teacher forcing
        transformer_out_list = self.transformer(dgcnn_out, num_stages, teacher_forcing)
        
        # Always compute the final output using Transform_Head
        transforms_sequence = self.transform_head(dgcnn_out)  # Shape: (batch_size, max_stages, num_teeth, 6)
        
        # Pad or truncate to max_stages (optional, depending on transformer's output)
        if transforms_sequence.size(1) < self.max_stages:
            padding = torch.zeros(
                batch_size, self.max_stages - transforms_sequence.size(1), self.num_teeth, 6,
                device=transforms_sequence.device
            )
            transforms_sequence = torch.cat([transforms_sequence, padding], dim=1)
        elif transforms_sequence.size(1) > self.max_stages:
            transforms_sequence = transforms_sequence[:, :self.max_stages, :, :]
        
        return transforms_sequence, num_stages_pred