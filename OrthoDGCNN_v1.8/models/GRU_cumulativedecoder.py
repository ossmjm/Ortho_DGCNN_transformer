import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class GRUToothDecoder(nn.Module):
    def __init__(self, d_model=36, hidden_size=128, num_layers=1, dim_feedforward=64, dropout=0.4):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Layer normalization for input features
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

        # Embeddings for teacher forcing
        self.cumulative_embed = nn.Linear(6, d_model)  # Embed previous transformations

        # Attention for neighbor weighting
        self.neighbor_attention = nn.MultiheadAttention(d_model, num_heads=2, dropout=dropout, batch_first=True)

        self.lstm = nn.LSTM(
            input_size=d_model * 3,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.0,
            batch_first=True
        )
        self.lstm_norm = nn.LayerNorm(hidden_size, eps=1e-6)


        # Translation prediction head
        self.trans_mlp = nn.Sequential(
            nn.Linear(hidden_size, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 3)  # Left/Right, Forward/Backward, Extrude/Intrude
        )

        # Rotation prediction head
        self.rot_mlp = nn.Sequential(
            nn.Linear(hidden_size, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 3)  # Buccal/Lingual, Mesial/Distal, Rotation
        )

        # Active/Inactive classification head
        self.active_mlp = nn.Sequential(
            nn.Linear(hidden_size, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 6),  # Binary classification for 6 parameters (all transformations)
            nn.Sigmoid()
        )

        # Direction classification head
        self.direction_mlp = nn.Sequential(
            nn.Linear(hidden_size, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 6),  # Binary classification for 6 parameters
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)

    def _get_teacher_forcing_params(self, epoch, total_epochs, tooth_idx, val_loss=None, base_tf_prob=0.95):
        logger = logging.getLogger('TrainLogger')
        min_tf_prob = 0.3
        progress = epoch / total_epochs
        tf_prob = max(min_tf_prob, base_tf_prob - 0.6 * progress)
        
        logger.debug(f"Teacher forcing prob: epoch={epoch}, tooth_idx={tooth_idx}, val_loss={val_loss if val_loss is not None else 'None'}, tf_prob={tf_prob:.4f}")
        return tf_prob

    def _aggregate_neighbor_features(self, x, batch_size, num_teeth):
        """Aggregate neighbor and global features, emphasizing neighbors via attention based on dental arch order."""
        d_model = x.shape[-1]
        neighbor_features = torch.zeros(batch_size, num_teeth, d_model, device=x.device)
        
        # Define tooth order: 37 → 36 → 35 → 34 → 33 → 32 → 31 → 41 → 42 → 43 → 44 → 45 → 46 → 47
        tooth_order = [37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47]
        neighbor_mask = torch.zeros(batch_size, num_teeth, num_teeth, device=x.device)
        for idx, tooth in enumerate(tooth_order):
            neighbors = []
            if idx > 0:
                neighbors.append(idx - 1)  # Left neighbor
            if idx < num_teeth - 1:
                neighbors.append(idx + 1)  # Right neighbor
            if neighbors:
                neighbor_mask[:, idx, neighbors] = 1.0
        
        queries = x[:, :, :].unsqueeze(2)  # (batch_size, num_teeth, 1, d_model)
        keys_values = x[:, :, :].unsqueeze(1).expand(-1, num_teeth, -1, -1)  # (batch_size, num_teeth, num_teeth, d_model)
        attn_output, _ = self.neighbor_attention(
            queries.view(batch_size * num_teeth, 1, d_model),
            keys_values.reshape(batch_size * num_teeth, num_teeth, d_model),
            keys_values.reshape(batch_size * num_teeth, num_teeth, d_model),
            key_padding_mask=~neighbor_mask.view(batch_size * num_teeth, num_teeth).bool()
        )
        neighbor_features = attn_output.view(batch_size, num_teeth, d_model)
        
        global_context = x.mean(dim=1, keepdim=True).expand(-1, num_teeth, -1)
        combined_context = 0.7 * neighbor_features + 0.3 * global_context
        
        return combined_context

    def forward(self, x, cumulative_transforms=None, active_labels=None, direction_labels=None, training=False, epoch=0, total_epochs=100, val_loss=None, is_freeze='none'):
        """
        x: (batch_size, num_teeth=14, d_model=36) - per-tooth features
        cumulative_transforms: (batch_size, num_teeth=14, 6) - ground-truth transformations (optional for teacher forcing)
        active_labels: (batch_size, num_teeth=14, 6) - binary labels (1=active, 0=inactive) for all transformations
        direction_labels: (batch_size, num_teeth=14, 6) - binary labels (1=positive, 0=negative)
        is_freeze: 'regression', 'classification', or 'none' - freeze opposite task to train specified task
        """
        logger = logging.getLogger('TrainLogger')
        if torch.isnan(x).any():
            logger.error("NaN values detected in input features")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        # Normalize input features
        batch_size, num_teeth, d_model = x.shape
        x = self.norm(x)  # (batch_size, 14, d_model)
        context = self._aggregate_neighbor_features(x, batch_size, num_teeth)  # (batch_size, 14, d_model)

        # Initialize outputs
        trans_mag = torch.zeros(batch_size, num_teeth, 3, device=x.device)
        rot_mag = torch.zeros(batch_size, num_teeth, 3, device=x.device)
        active_pred = torch.zeros(batch_size, num_teeth, 6, device=x.device)  # Changed to [batch_size, num_teeth, 6]
        direction_pred = torch.zeros(batch_size, num_teeth, 6, device=x.device)

        # Freeze specified task
        if is_freeze == 'regression':
            # Freeze classification heads, train regression heads
            self.trans_mlp.train()
            self.rot_mlp.train()
            self.active_mlp.eval()
            self.direction_mlp.eval()
            for param in self.trans_mlp.parameters():
                param.requires_grad = True
            for param in self.rot_mlp.parameters():
                param.requires_grad = True
            for param in self.active_mlp.parameters():
                param.requires_grad = False
            for param in self.direction_mlp.parameters():
                param.requires_grad = False
        elif is_freeze == 'classification':
            # Freeze regression heads, train classification heads
            self.trans_mlp.eval()
            self.rot_mlp.eval()
            self.active_mlp.train()
            self.direction_mlp.train()
            for param in self.trans_mlp.parameters():
                param.requires_grad = False
            for param in self.rot_mlp.parameters():
                param.requires_grad = False
            for param in self.active_mlp.parameters():
                param.requires_grad = True
            for param in self.direction_mlp.parameters():
                param.requires_grad = True
        else:
            # Train all heads
            self.trans_mlp.train()
            self.rot_mlp.train()
            self.active_mlp.train()
            self.direction_mlp.train()
            for param in self.trans_mlp.parameters():
                param.requires_grad = True
            for param in self.rot_mlp.parameters():
                param.requires_grad = True
            for param in self.active_mlp.parameters():
                param.requires_grad = True
            for param in self.direction_mlp.parameters():
                param.requires_grad = True

        # Initialize GRU hidden state
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size,device=x.device)

        # Process teeth sequentially
        prev_trans = []
        prev_rot = []
        for tooth_idx in range(num_teeth):
            tf_prob = self._get_teacher_forcing_params(epoch, total_epochs, tooth_idx, val_loss)
            effective_tf_prob = tf_prob if training else 0.0
            use_tf = training and torch.rand(1).item() < effective_tf_prob and cumulative_transforms is not None

            # Prepare input
            tooth_features = x[:, tooth_idx, :].unsqueeze(1)  # (batch_size, 1, d_model)
            context_features = context[:, tooth_idx, :].unsqueeze(1)  # (batch_size, 1, d_model)

            # Teacher forcing or previous predictions
            if use_tf:
                trans_prev = cumulative_transforms[:, tooth_idx, :3].unsqueeze(1)
                rot_prev = cumulative_transforms[:, tooth_idx, 3:].unsqueeze(1)
            else:
                trans_prev = prev_trans[-1].unsqueeze(1) if prev_trans else torch.zeros(batch_size, 1, 3, device=x.device)
                rot_prev = prev_rot[-1].unsqueeze(1) if prev_rot else torch.zeros(batch_size, 1, 3, device=x.device)

            # Embed previous transformations
            cumulative_input = self.cumulative_embed(torch.cat([trans_prev, rot_prev], dim=-1))  # (batch_size, 1, d_model)

            # GRU input
            lstm_input = torch.cat([tooth_features, context_features, cumulative_input], dim=-1)  # (batch_size, 1, d_model * 3)
            lstm_output, (hidden, c_0) = self.lstm(lstm_input, (hidden,c_0))
            lstm_output = lstm_output.squeeze(1)  # (batch_size, hidden_size)
            lstm_output = self.lstm_norm(lstm_output)
            lstm_output = F.dropout(lstm_output, p=self.dropout, training=self.training)

            # Predictions
            trans_pred = self.trans_mlp(lstm_output)  # (batch_size, 3)
            trans_mag[:, tooth_idx, :] = trans_pred

            rot_pred = self.rot_mlp(lstm_output)  # (batch_size, 3)
            rot_mag[:, tooth_idx, :] = rot_pred

            active_pred[:, tooth_idx, :] = self.active_mlp(lstm_output)  # (batch_size, 6)

            direction_pred[:, tooth_idx, :] = self.direction_mlp(lstm_output)  # (batch_size, 6)

            # Update previous predictions
            prev_trans.append(trans_pred.detach())
            prev_rot.append(rot_pred.detach())

        # Check for NaN
        for output, name in [(trans_mag, 'trans_mag'), (rot_mag, 'rot_mag'), (active_pred, 'active_pred'), (direction_pred, 'direction_pred')]:
            if torch.isnan(output).any():
                logger.error(f"NaN values detected in {name}")
                output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)

        return trans_mag, rot_mag, active_pred, direction_pred