import torch
import torch.nn as nn
import logging

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
            dropout=0.3,
            batch_first=True
        )
        self.post_transformer_ffn = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.out_layer = nn.Linear(d_model, self.num_teeth * 6)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_stages, d_model))  # Zero-initialized
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, dgcnn_out, targets=None, teacher_forcing=False, epoch=None, total_epochs=None):
        batch_size = dgcnn_out.size(0)
        src = dgcnn_out.unsqueeze(1).repeat(1, self.max_stages, 1)
        src = src + self.positional_encoding[:, :self.max_stages, :]
        
        logger = logging.getLogger('TrainLogger')
        
        if self.training and teacher_forcing and targets is not None:
            alpha = min(1.0, epoch / (total_epochs * 0.75)) if epoch is not None and total_epochs is not None else 1.0
            if torch.rand(1).item() < alpha:
                tgt = targets.view(batch_size, self.max_stages, -1)
                tgt = torch.nn.functional.pad(tgt, (0, self.d_model - self.num_teeth * 6))
                start_token = torch.zeros(batch_size, 1, self.d_model, device=dgcnn_out.device)
                tgt = torch.cat([start_token, tgt[:, :-1, :]], dim=1)
                tgt = tgt + self.positional_encoding[:, :self.max_stages, :]
                
                transformer_out = self.transformer(src, tgt)
                transformer_out = self.post_transformer_ffn(transformer_out)
                transformer_out = self.out_layer(transformer_out)
            else:
                tgt = torch.zeros(batch_size, 1, self.d_model, device=dgcnn_out.device)
                outputs = []
                for t in range(self.max_stages):
                    tgt_t = tgt + self.positional_encoding[:, t:t+1, :]
                    transformer_out = self.transformer(src[:, :t+1, :], tgt_t)
                    transformer_out = self.post_transformer_ffn(transformer_out)
                    stage_out = self.out_layer(transformer_out[:, -1:, :])
                    outputs.append(stage_out)
                    next_tgt = torch.nn.functional.pad(stage_out, (0, self.d_model - self.num_teeth * 6))
                    tgt = torch.cat([tgt, next_tgt], dim=1)
                
                transformer_out = torch.cat(outputs, dim=1)
        else:
            tgt = torch.zeros(batch_size, 1, self.d_model, device=dgcnn_out.device)
            outputs = []
            for t in range(self.max_stages):
                tgt_t = tgt + self.positional_encoding[:, t:t+1, :]
                transformer_out = self.transformer(src[:, :t+1, :], tgt_t)
                transformer_out = self.post_transformer_ffn(transformer_out)
                stage_out = self.out_layer(transformer_out[:, -1:, :])
                outputs.append(stage_out)
                next_tgt = torch.nn.functional.pad(stage_out, (0, self.d_model - self.num_teeth * 6))
                tgt = torch.cat([tgt, next_tgt], dim=1)
            
            transformer_out = torch.cat(outputs, dim=1)
        
        transformer_out = torch.clamp(transformer_out, -1.0, 1.0)  # Match normalized input range
        logger.debug(f"Transformer output range: min={transformer_out.min().item():.4f}, max={transformer_out.max().item():.4f}")
        
        return transformer_out