# StageTransformer.ipynb
import torch
import torch.nn as nn

class StageTransformer(nn.Module):
    def __init__(self, num_teeth=14, num_features=512, d_model=7168, num_heads=8, num_layers=4, max_stages=20):
        super(StageTransformer, self).__init__()
        self.num_teeth = num_teeth
        self.d_model = d_model
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_stages, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, num_stages, teacher_forcing=None):
        transformer_out_list = []
        batch_size = x.shape[0]
        for b in range(batch_size):
            n_stages = num_stages[b].item()  # Predicted or true num_stages
            x_b = x[b:b+1].repeat(n_stages, 1)  # (n_stages, 7168)
            if teacher_forcing is not None and self.training:
                # Teacher forcing: Replace all but last stage with ground truth
                x_b[:n_stages-1] = teacher_forcing[b, :n_stages-1].view(n_stages-1, -1)
            x_b = x_b.unsqueeze(0) + self.pos_encoding[:, :n_stages, :]  # (1, n_stages, 7168)
            transformer_out = self.transformer(x_b)  # (1, n_stages, 7168)
            transformer_out_list.append(transformer_out.squeeze(0))  # (n_stages, 7168)
        return transformer_out_list