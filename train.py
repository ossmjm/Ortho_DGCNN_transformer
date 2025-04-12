# train_model.py
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
from dataset import JawTeethDataset
from models.DGCNN import DGCNN
from models.StageTransformer import StageTransformer
from models.OrthoDGCNN import OrthoDGCNNModel

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset = JawTeethDataset(args.data_dir, max_stages=args.max_stages, split='train', train_ratio=args.train_ratio, inference=False)
    test_dataset = JawTeethDataset(args.data_dir, max_stages=args.max_stages, split='test', train_ratio=args.train_ratio, inference=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    dgcnn = DGCNN(in_channels=3, embed_dim=512, num_teeth=14, k=20).to(device)
    transformer = StageTransformer(d_model=14 * 512, max_stages=args.max_stages).to(device)
    model = OrthoDGCNNModel(dgcnn, transformer, max_stages=args.max_stages).to(device)
    
    optimizer = torch.optim.Adam(
        list(dgcnn.parameters()) + list(transformer.parameters()) + 
        list(model.stage_predictor.parameters()) + list(model.transform_head.parameters()), 
        lr=args.lr
    )
    transform_criterion = nn.MSELoss()
    stages_criterion = nn.MSELoss()
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            cordinates, targets, true_num_stages = batch
            cordinates, targets, true_num_stages = [
                x.to(device) for x in [cordinates, targets, true_num_stages]
            ]
            
            optimizer.zero_grad()
            transforms_sequence, num_stages_pred = model(
                cordinates, teacher_forcing=targets, true_num_stages=true_num_stages,
                epoch=epoch, total_epochs=args.epochs
            )
            
            transform_loss = transform_criterion(transforms_sequence, targets)
            stages_loss = stages_criterion(num_stages_pred.float(), true_num_stages.float())
            loss = transform_loss + args.stages_loss_weight * stages_loss
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss / len(train_loader):.4f}")
        
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                cordinates, targets, true_num_stages = batch
                cordinates, targets, true_num_stages = [
                    x.to(device) for x in [cordinates, targets, true_num_stages]
                ]
                transforms_sequence, num_stages_pred = model(cordinates)
                max_stages = num_stages_pred.max().item()
                transform_loss = transform_criterion(
                    transforms_sequence[:, :max_stages, :, :], targets[:, :max_stages, :, :]
                )
                stages_loss = stages_criterion(num_stages_pred.float(), true_num_stages.float())
                loss = transform_loss + args.stages_loss_weight * stages_loss
                test_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs}, Test Loss: {test_loss / len(test_loader):.4f}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(dgcnn.state_dict(), os.path.join(args.output_dir, "dgcnn.pth"))
    torch.save(model.stage_predictor.state_dict(), os.path.join(args.output_dir, "stage_predictor.pth"))
    torch.save(transformer.state_dict(), os.path.join(args.output_dir, "stage_transformer.pth"))
    torch.save(model.state_dict(), os.path.join(args.output_dir, "ortho_dgcnn.pth"))

if __name__ == "__main__":
    class Args:
        data_dir = "/media/osama/sm/Sample_data"
        max_stages = 20
        train_ratio = 0.8
        output_dir = "output"
        lr = 0.001
        epochs = 10
        batch_size = 1
        stages_loss_weight = 1.0
    args = Args()
    train_model(args)