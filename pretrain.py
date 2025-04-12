# pretrain_fused_encoder.ipynb
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

def pretrain_fused_encoder(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = JawTeethDataset(args.data_dir, max_stages=args.max_stages, split='train', train_ratio=args.train_ratio)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    fused_encoder = FusedEncoder().to(device)
    optimizer = torch.optim.Adam(fused_encoder.parameters(), lr=args.lr)
    criterion = nn.MSELoss()  # Simplified: predict flattened transformations as pretext task
    
    for epoch in range(args.epochs):
        fused_encoder.train()
        train_loss = 0
        for batch in dataloader:
            faces, feats, centers, Fs, cordinates, targets, _, all_vertices, _ = batch
            faces, feats, centers, Fs, cordinates, targets, all_vertices = [x.to(device) for x in [faces, feats, centers, Fs, cordinates, targets, all_vertices]]
            
            optimizer.zero_grad()
            features = fused_encoder(faces, feats, centers, Fs, cordinates, all_vertices)  # (batch_size, 512)
            
            # Pretext task: Predict flattened first-stage transformations
            target_transforms = targets[:, 0, :, :].reshape(targets.shape[0], -1)  # (batch_size, 14 * 6)
            pred_transforms = nn.Linear(512, 14 * 6).to(device)(features)  # Simplified head for pretraining
            loss = criterion(pred_transforms, target_transforms)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Pretrain Epoch {epoch+1}, Loss: {train_loss / len(dataloader)}")
    
    torch.save(fused_encoder.state_dict(), args.checkpoint_path)

# Example args
class Args:
    data_dir = "data"
    max_stages = 20
    train_ratio = 0.8
    lr = 0.001
    epochs = 50
    batch_size = 2
    checkpoint_path = "fused_encoder.pth"

args = Args()
pretrain_fused_encoder(args)