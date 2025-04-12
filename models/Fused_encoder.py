# FusedEncoder.ipynb
import torch
import torch.nn as nn

class MeshEncoder(nn.Module):
    def __init__(self):
        super(MeshEncoder, self).__init__()
        self.conv = nn.Conv1d(3, 512, 1)  # Simplified for demo; adjust as needed
        self.bn = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class FusedEncoder(nn.Module):
    def __init__(self, num_teeth=14):
        super(FusedEncoder, self).__init__()
        self.mesh_encoder = MeshEncoder()
        self.num_teeth = num_teeth
        self.tooth_processor = nn.Sequential(
            nn.Linear(512 * num_teeth, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)  # Single fused output
        )

    def forward(self, faces, feats, centers, Fs, cordinates, all_vertices):
        # Process tooth-specific features
        batch_size = cordinates.shape[0]
        tooth_features = self.mesh_encoder(cordinates.mean(dim=2).transpose(1, 2))  # (batch_size, 512, 14)
        tooth_features = tooth_features.transpose(1, 2).reshape(batch_size, -1)  # (batch_size, 512 * 14)
        
        # Process jaw-wide vertices (simplified for fusion)
        global_features = self.mesh_encoder(all_vertices.transpose(1, 2)).max(dim=2)[0]  # (batch_size, 512)
        
        # Fuse local and global features
        fused_features = torch.cat([tooth_features, global_features], dim=1)  # (batch_size, 512 * 14 + 512)
        fused_output = self.tooth_processor(fused_features)  # (batch_size, 512)
        
        return fused_output  # Single output combining local and global info