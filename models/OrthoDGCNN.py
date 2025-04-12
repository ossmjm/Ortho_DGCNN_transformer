# OrthoDGCNNModel.ipynb
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.StageTransformer import StageTransformer
from models.Fused_encoder import FusedEncoder
from models.StagePredictor import StagePredictor

class OrthoDGCNNModel(nn.Module):
    def __init__(self, dgcnn, transformer, max_stages=20):
        super(OrthoDGCNNModel, self).__init__()
        self.dgcnn = dgcnn
        self.transformer = transformer
        self.stage_predictor = StagePredictor(in_features=7168, hidden_features=512, max_stages=max_stages)
        self.transform_head = TransformHead(num_teeth=14)
        self.max_stages = max_stages
        self.num_teeth = 14

    def forward(self, cordinates,teacher_forcing=None, true_num_stages=None, epoch=None, total_epochs=100):
        # DGCNN processes raw coordinates directly
        dgcnn_out = self.dgcnn(cordinates)  # (batch_size, 7168)
        
        num_stages_pred = self.stage_predictor(dgcnn_out)
        p_tf = max(0.0, 1.0 - (epoch / (total_epochs * 0.5))) if self.training and epoch is not None else 0.0
        use_true = self.training and true_num_stages is not None and torch.rand(1).item() < p_tf
        num_stages = true_num_stages if use_true else num_stages_pred
        
        transformer_out_list = self.transformer(dgcnn_out, num_stages, teacher_forcing)
        
        transforms_sequence = []
        for b in range(dgcnn_out.shape[0]):
            transforms = self.transform_head(transformer_out_list[b])
            transforms = transforms.view(num_stages[b], self.num_teeth, 6)
            transforms_sequence.append(transforms)
        
        if self.training:
            padded_transforms = torch.zeros(dgcnn_out.shape[0], self.max_stages, self.num_teeth, 6).to(dgcnn_out.device)
            for b in range(dgcnn_out.shape[0]):
                padded_transforms[b, :num_stages[b], :, :] = transforms_sequence[b]
            return padded_transforms, num_stages_pred
        return torch.stack(transforms_sequence), num_stages_pred
    

class TransformHead(nn.Module):
    def __init__(self, num_teeth=14, in_features=7168, hidden_features=512):
        super(TransformHead, self).__init__()
        self.num_teeth = num_teeth
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_teeth * 6)  # 6 params per tooth

    def forward(self, x):
        x = F.relu(self.fc1(x))  # (n_stages, 512)
        x = self.fc2(x)  # (n_stages, num_teeth * 6)
        return x