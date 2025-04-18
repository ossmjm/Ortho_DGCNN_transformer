import torch
import torch.nn as nn

class StageTransformer(nn.Module):
    def __init__(self, d_model, max_stages, n_head, num_encoder_layers, num_decoder_layers):
        super(StageTransformer, self).__init__()
        self.d_model = d_model
        self.max_stages = max_stages
        self.num_teeth = 14
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_head,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
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
        self.positional_encoding = nn.Parameter(torch.randn(1, max_stages, d_model))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, dgcnn_out, teacher_forcing=None, epoch=None, total_epochs=None):
        batch_size = dgcnn_out.size(0)
        src = dgcnn_out.unsqueeze(1).repeat(1, self.max_stages, 1)  # Shape: [batch_size, max_stages, d_model]
        src = src + self.positional_encoding[:, :self.max_stages, :]

        if self.training and teacher_forcing is not None:
            # Scheduled teacher forcing
            alpha = min(1.0, epoch / (total_epochs * 0.5)) if epoch is not None and total_epochs is not None else 1.0
            if torch.rand(1).item() < alpha:
                # Teacher forcing
                tgt = teacher_forcing.view(batch_size, self.max_stages, -1)  # Shape: [batch_size, max_stages, num_teeth*6]
                tgt = torch.nn.functional.pad(tgt, (0, self.d_model - self.num_teeth * 6))  # Shape: [batch_size, max_stages, d_model]
                start_token = torch.zeros(batch_size, 1, self.d_model, device=dgcnn_out.device)  # Shape: [batch_size, 1, d_model]
                tgt = torch.cat([start_token, tgt[:, :-1, :]], dim=1)  # Shape: [batch_size, max_stages, d_model]
                tgt = tgt + self.positional_encoding[:, :self.max_stages, :]
                
                transformer_out = self.transformer(src, tgt)  # Shape: [batch_size, max_stages, d_model]
                transformer_out = self.post_transformer_ffn(transformer_out)  # Shape: [batch_size, max_stages, d_model]
                transformer_out = self.out_layer(transformer_out)  # Shape: [batch_size, max_stages, num_teeth*6]
            else:
                # Autoregressive training
                tgt = torch.zeros(batch_size, 1, self.d_model, device=dgcnn_out.device)  # Shape: [batch_size, 1, d_model]
                outputs = []
                for t in range(self.max_stages):
                    tgt_t = tgt + self.positional_encoding[:, t:t+1, :]
                    transformer_out = self.transformer(src[:, :t+1, :], tgt_t)  # Shape: [batch_size, t+1, d_model]
                    transformer_out = self.post_transformer_ffn(transformer_out)  # Shape: [batch_size, t+1, d_model]
                    stage_out = self.out_layer(transformer_out[:, -1:, :])  # Shape: [batch_size, 1, num_teeth*6]
                    outputs.append(stage_out)
                    next_tgt = torch.nn.functional.pad(stage_out, (0, self.d_model - self.num_teeth * 6))  # Shape: [batch_size, 1, d_model]
                    tgt = torch.cat([tgt, next_tgt], dim=1)  # Shape: [batch_size, t+2, d_model]
                
                transformer_out = torch.cat(outputs, dim=1)  # Shape: [batch_size, max_stages, num_teeth*6]
        else:
            # Inference: autoregressive generation
            tgt = torch.zeros(batch_size, 1, self.d_model, device=dgcnn_out.device)  # Shape: [batch_size, 1, d_model]
            outputs = []
            for t in range(self.max_stages):
                tgt_t = tgt + self.positional_encoding[:, t:t+1, :]
                transformer_out = self.transformer(src[:, :t+1, :], tgt_t)  # Shape: [batch_size, t+1, d_model]
                transformer_out = self.post_transformer_ffn(transformer_out)  # Shape: [batch_size, t+1, d_model]
                stage_out = self.out_layer(transformer_out[:, -1:, :])  # Shape: [batch_size, 1, num_teeth*6]
                outputs.append(stage_out)
                next_tgt = torch.nn.functional.pad(stage_out, (0, self.d_model - self.num_teeth * 6))  # Shape: [batch_size, 1, d_model]
                tgt = torch.cat([tgt, next_tgt], dim=1)  # Shape: [batch_size, t+2, d_model]
            
            transformer_out = torch.cat(outputs, dim=1)  # Shape: [batch_size, max_stages, num_teeth*6]
        
        return transformer_out