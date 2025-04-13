# models/StagePredictor.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class StagePredictor(nn.Module):
    def __init__(self, in_features, hidden_features, max_stages):
        super(StagePredictor, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, 1)
        self.max_stages = max_stages

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.clamp(x, 1, self.max_stages).round()