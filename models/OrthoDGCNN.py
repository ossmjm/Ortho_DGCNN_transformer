# models/OrthoDGCNN.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.StagePredictor import StagePredictor

class OrthoDGCNNModel(nn.Module):
    def __init__(self, dgcnn, transformer, max_stages=20, num_teeth=14, embed_dim=256):
        super(OrthoDGCNNModel, self).__init__()
        self.dgcnn = dgcnn
        self.transformer = transformer
        self.stage_predictor = StagePredictor(in_features=num_teeth * embed_dim, hidden_features=256, max_stages=max_stages)
        self.transform_head = TransformHead(num_teeth=num_teeth)
        self.max_stages = max_stages
        self.num_teeth = num_teeth

    def forward(self, cordinates, teacher_forcing=None, true_num_stages=None, epoch=None, total_epochs=100):
        dgcnn_out = self.dgcnn(cordinates)
        
        num_stages_pred = self.stage_predictor(dgcnn_out)
        p_tf = max(0.0, 1.0 - (epoch / (total_epochs * 0.5))) if self.training and epoch is not None else 0.0
        use_true = self.training and true_num_stages is not None and torch.rand(1).item() < p_tf
        num_stages = true_num_stages if use_true else num_stages_pred
        num_stages = num_stages.long()  # Fix: Convert to integer type for indexing
        
        transformer_out_list = self.transformer(dgcnn_out, num_stages, teacher_forcing)
        
        transforms_sequence = []
        for b in range(dgcnn_out.shape[0]):
            transformer_out = transformer_out_list[b].view(num_stages[b], -1)
            transforms = self.transform_head(transformer_out)
            transforms = transforms.view(num_stages[b], self.num_teeth, 6)
            transforms_sequence.append(transforms)
        
        if self.training:
            padded_transforms = torch.zeros(dgcnn_out.shape[0], self.max_stages, self.num_teeth, 6).to(dgcnn_out.device)
            for b in range(dgcnn_out.shape[0]):
                padded_transforms[b, :num_stages[b], :, :] = transforms_sequence[b]
            return padded_transforms, num_stages_pred
        return torch.stack(transforms_sequence), num_stages_pred

class TransformHead(nn.Module):
    def __init__(self, num_teeth=14, hidden_features=256):
        super(TransformHead, self).__init__()
        self.num_teeth = num_teeth
        self.fc1 = nn.Linear(num_teeth * 6, hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_teeth * 6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x