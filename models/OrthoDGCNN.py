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
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size * self.max_stages, self.in_dim)
        out = self.layers(x)
        out = out.view(batch_size, self.max_stages, self.num_teeth, 6)
        return out

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
        dgcnn_out = self.dgcnn(cordinates)  # Shape: [batch_size, num_teeth * embed_dim]
        transformer_out = self.transformer(
            dgcnn_out, 
            targets=targets, 
            teacher_forcing=self.teacher_forcing, 
            epoch=epoch, 
            total_epochs=total_epochs
        )  # Shape: [batch_size, max_stages, num_teeth*6]
        transforms_sequence = self.transform_head(transformer_out)  # Shape: [batch_size, max_stages, num_teeth, 6]
        return transforms_sequence