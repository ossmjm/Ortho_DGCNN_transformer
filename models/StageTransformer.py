# models/StageTransformer.py
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
            dropout=0.1
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

    def forward(self, dgcnn_out, num_stages, teacher_forcing=None):
        batch_size = dgcnn_out.size(0)
        # print(f"StageTransformer: num_stages shape = {num_stages.shape}, values = {num_stages.tolist()}")
        
        transformer_out_list = []
        for b in range(batch_size):
            stages = num_stages[b].item()
            print(f"StageTransformer: batch {b}, stages = {stages}")
            
            # Source sequence: repeat dgcnn_out for each stage
            src = dgcnn_out[b:b+1].repeat(stages, 1)  # Shape: (stages, d_model)

            if teacher_forcing is not None and self.training:
                # Teacher forcing: use ground truth transformations for stages 1 to t-1 to predict stage t
                # Prepare tgt: prepend a zero vector (start token) and use ground truth up to t-1
                tgt = teacher_forcing[b, :stages, :, :]  # Shape: (stages, num_teeth, 6)
                tgt = tgt.view(stages, -1)  # Shape: (stages, num_teeth * 6)
                tgt = torch.nn.functional.pad(tgt, (0, self.d_model - self.num_teeth * 6))  # Shape: (stages, d_model)
                
                # Prepend a zero vector (start token) to tgt
                start_token = torch.zeros(1, self.d_model, device=dgcnn_out.device)  # Shape: (1, d_model)
                tgt = torch.cat([start_token, tgt[:-1, :]], dim=0)  # Shape: (stages, d_model)
                # Now, tgt[t-1] contains the ground truth for stage t-1, and the transformer will predict stage t
            else:
                # Inference: autoregressive generation
                tgt = torch.zeros(1, self.d_model, device=dgcnn_out.device)  # Start with a zero vector
                outputs = []
                for t in range(stages):
                    # Predict the next stage
                    transformer_out = self.transformer(src[:t+1, :], tgt)  # Shape: (t+1, d_model)
                    transformer_out = self.post_transformer_ffn(transformer_out)  # Shape: (t+1, d_model)
                    stage_out = self.out_layer(transformer_out[-1:])  # Shape: (1, num_teeth * 6)
                    
                    # Convert the output to the next tgt input
                    next_tgt = torch.nn.functional.pad(stage_out, (0, self.d_model - self.num_teeth * 6))  # Shape: (1, d_model)
                    tgt = torch.cat([tgt, next_tgt], dim=0)  # Shape: (t+2, d_model)
                    outputs.append(stage_out)
                
                transformer_out = torch.cat(outputs, dim=0)  # Shape: (stages, num_teeth * 6)
                # Skip the transformer call below since we've already computed the output
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
                continue

            # Training with teacher forcing: predict all stages at once
            transformer_out = self.transformer(src, tgt)  # Shape: (stages, d_model)
            transformer_out = self.post_transformer_ffn(transformer_out)  # Shape: (stages, d_model)
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