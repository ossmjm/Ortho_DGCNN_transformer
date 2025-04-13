# models/StageTransformer.py
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential

class StageTransformer(nn.Module):
    def __init__(self, d_model, max_stages=20):
        super(StageTransformer, self).__init__()
        self.d_model = d_model
        self.max_stages = max_stages
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            batch_first=True,
            dropout=0.5
        )
        self.fc = nn.Linear(d_model, 14 * 6)
        self.teacher_forcing_projection = nn.Linear(14 * 6, d_model)

    def forward(self, dgcnn_out, num_stages, teacher_forcing=None):
        batch_size = dgcnn_out.size(0)
        transformer_out_list = []

        for b in range(batch_size):
            n_stages = num_stages[b].item() if num_stages is not None else self.max_stages
            x_b = torch.zeros(self.max_stages, self.d_model, device=dgcnn_out.device)

            if teacher_forcing is not None and self.training:
                teacher_forcing_flat = teacher_forcing[b, :n_stages-1].view(n_stages-1, -1)
                teacher_forcing_projected = self.teacher_forcing_projection(teacher_forcing_flat)
                x_b = torch.cat((teacher_forcing_projected, x_b[n_stages-1:]), dim=0)

            x_b_list = [x_b[i] for i in range(self.max_stages)]

            for t in range(n_stages):
                src = dgcnn_out[b].unsqueeze(0).repeat(t + 1, 1)
                tgt = torch.stack(x_b_list[:t + 1])
                src = src.unsqueeze(0)
                tgt = tgt.unsqueeze(0)
                transformer_out = checkpoint_sequential(
                    modules=[self.transformer.encoder, self.transformer.decoder],
                    segments=2,
                    input=(src, tgt)
                )
                decoder_out = transformer_out[1]  # Decoder output
                x_b_list[t] = decoder_out[:, -1, :]  # Shape: (1, d_model)

            x_b_updated = torch.stack(x_b_list)
            transformer_out = self.fc(x_b_updated[:n_stages])
            transformer_out_list.append(transformer_out)

        transformer_out = torch.stack(transformer_out_list)
        return transformer_out.view(batch_size, -1, 14, 6)