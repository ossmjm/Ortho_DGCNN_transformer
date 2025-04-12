# stage_predictor.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class StagePredictor(nn.Module):
    def __init__(self, in_features=7168, hidden_features=512, max_stages=20):
        super(StagePredictor, self).__init__()
        self.max_stages = max_stages
        
        # Layers inspired by TransformHead to process DGCNN output
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.fc2 = nn.Linear(hidden_features, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 1)  # Output single num_stages value
    
    def forward(self, x):
        # x: (batch_size, 7168) from DGCNN
        x = F.relu(self.bn1(self.fc1(x)))  # (batch_size, 512)
        x = F.relu(self.bn2(self.fc2(x)))  # (batch_size, 256)
        stage_logits = self.fc3(x)  # (batch_size, 1)
        num_stages = torch.clamp(stage_logits, min=1, max=self.max_stages).round().int()  # (batch_size,)
        return num_stages