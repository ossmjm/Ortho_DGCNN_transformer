# models/StageTransformer.py
import torch
import torch.nn as nn

class StageTransformer(nn.Module):
    def __init__(self, d_model, max_stages, n_head, num_encoder_layers, num_decoder_layers):
        super(StageTransformer, self).__init__()
        self.d_model = d_model  # e.g., num_teeth * embed_dim = 14 * 128 = 1792
        self.max_stages = max_stages
        self.num_teeth = 14
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_head,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=2048,
            dropout=0.1
        )
        # Feedforward network to refine transformer output
        self.post_transformer_ffn = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.out_layer = nn.Linear(d_model, self.num_teeth * 6)

    def forward(self, dgcnn_out, num_stages, teacher_forcing=None):
        batch_size = dgcnn_out.size(0)
        transformer_out_list = []

        for b in range(batch_size):
            stages = num_stages[b].item()
            src = dgcnn_out[b:b+1].repeat(stages, 1)
            if teacher_forcing is not None and self.training:
                tgt = teacher_forcing[b, :stages, :, :]
                tgt = tgt.view(stages, -1)
                tgt = torch.nn.functional.pad(tgt, (0, self.d_model - self.num_teeth * 6))
            else:
                tgt = torch.zeros(stages, self.d_model, device=dgcnn_out.device)

            transformer_out = self.transformer(src, tgt)  # Shape: (stages, d_model)
            transformer_out = self.post_transformer_ffn(transformer_out)  # Refine output
            transformer_out = self.out_layer(transformer_out)  # Shape: (stages, num_teeth * 6)

            if stages < self.max_stages:
                padding = torch.zeros(
                    self.max_stages - stages,
                    self.num_teeth * 6,
                    device=transformer_out.device
                )
                transformer_out = torch.cat([transformer_out, padding], dim=0)
            elif stages > self.max_stages:
                transformer_out = transformer_out[:self.max_stages, :]

            transformer_out_list.append(transformer_out)

        transformer_out = torch.stack(transformer_out_list)  # Shape: (batch_size, max_stages, num_teeth * 6)
        return transformer_out