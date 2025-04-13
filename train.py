# train.py
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import argparse
from dataset import JawTeethDataset
from models.DGCNN import DGCNN
from models.StageTransformer import StageTransformer
from models.OrthoDGCNN import OrthoDGCNNModel

# Set environment variables for CUDA
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.cuda.empty_cache()
    
    # Initialize datasets
    train_dataset = JawTeethDataset(
        args.data_dir, 
        max_stages=args.max_stages, 
        num_patches=args.num_patches, 
        patch_size=args.patch_size, 
        channels=13, 
        split='train', 
        train_ratio=args.train_ratio, 
        inference=False
    )
    test_dataset = JawTeethDataset(
        args.data_dir, 
        max_stages=args.max_stages, 
        num_patches=args.num_patches, 
        patch_size=args.patch_size, 
        channels=13, 
        split='test', 
        train_ratio=args.train_ratio, 
        inference=False
    )
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    
    # Initialize models
    dgcnn = DGCNN(in_channels=13, embed_dim=args.embed_dim, num_teeth=14, k=10).to(device)
    transformer = StageTransformer(d_model=14 * args.embed_dim, max_stages=args.max_stages).to(device)
    model = OrthoDGCNNModel(dgcnn, transformer, max_stages=args.max_stages, num_teeth=14, embed_dim=args.embed_dim).to(device)
    
    # Optimizer and loss functions
    optimizer = torch.optim.Adam(
        list(dgcnn.parameters()) + list(transformer.parameters()) + 
        list(model.stage_predictor.parameters()) + list(model.transform_head.parameters()), 
        lr=args.lr,
        weight_decay=1e-4
    )
    transform_criterion = nn.MSELoss()
    stages_criterion = nn.MSELoss()
    
    # Training utilities
    scaler = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    accumulation_steps = 2

    best_test_loss = float('inf')
    patience = 10
    patience_counter = 0

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, batch in enumerate(train_loader):
            cordinates, targets, true_num_stages = batch
            cordinates, targets, true_num_stages = [
                x.to(device) for x in [cordinates, targets, true_num_stages]
            ]
            
            with torch.amp.autocast('cuda'):
                transforms_sequence, num_stages_pred = model(
                    cordinates, teacher_forcing=targets, 
                    true_num_stages=true_num_stages,
                    epoch=epoch, total_epochs=args.epochs
                )
                transform_loss = transform_criterion(transforms_sequence, targets)
                stages_loss = stages_criterion(num_stages_pred.squeeze(-1).float(), true_num_stages.float())
                loss = transform_loss + args.stages_loss_weight * stages_loss
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            train_loss += loss.item() * accumulation_steps
        
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss / len(train_loader):.4f}")
        
        # Evaluation loop
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                cordinates, targets, true_num_stages = batch
                cordinates, targets, true_num_stages = [
                    x.to(device) for x in [cordinates, targets, true_num_stages]
                ]
                with torch.amp.autocast('cuda'):
                    transforms_sequence, num_stages_pred = model(cordinates)
                    max_stages = int(num_stages_pred.max().item())  # Convert to integer
                    max_stages = min(max(1, max_stages), args.max_stages)  # Ensure within bounds
                    transform_loss = transform_criterion(
                        transforms_sequence[:, :max_stages, :, :], targets[:, :max_stages, :, :]
                    )
                    stages_loss = stages_criterion(num_stages_pred.squeeze(-1).float(), true_num_stages.float())
                    loss = transform_loss + args.stages_loss_weight * stages_loss
                test_loss += loss.item()
        test_loss_avg = test_loss / len(test_loader)
        print(f"Epoch {epoch+1}/{args.epochs}, Test Loss: {test_loss_avg:.4f}")
        
        # Early stopping
        if test_loss_avg < best_test_loss:
            best_test_loss = test_loss_avg
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
        
        scheduler.step()

    # Save models
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(dgcnn.state_dict(), os.path.join(args.output_dir, "dgcnn.pth"))
    torch.save(model.stage_predictor.state_dict(), os.path.join(args.output_dir, "stage_predictor.pth"))
    torch.save(transformer.state_dict(), os.path.join(args.output_dir, "stage_transformer.pth"))
    torch.save(model.state_dict(), os.path.join(args.output_dir, "ortho_dgcnn.pth"))

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train an OrthoDGCNN model for orthodontic transformation prediction.")
    
    # Dataset and training arguments
    parser.add_argument('--data_dir', type=str, default="/media/osama/sm/Sample_data",
                        help="Path to the dataset directory containing jaw data.")
    parser.add_argument('--max_stages', type=int, default=20,
                        help="Maximum number of treatment stages.")
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help="Ratio of data to use for training (vs. testing).")
    parser.add_argument('--output_dir', type=str, default="output",
                        help="Directory to save model checkpoints and outputs.")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Learning rate for the optimizer.")
    parser.add_argument('--epochs', type=int, default=100,
                        help="Number of epochs to train the model.")
    parser.add_argument('--batch_size', type=int, default=2,
                        help="Batch size for training and evaluation.")
    parser.add_argument('--stages_loss_weight', type=float, default=1.0,
                        help="Weight for the stage prediction loss in the total loss.")
    
    # Model architecture arguments
    parser.add_argument('--embed_dim', type=int, default=256,
                        help="Embedding dimension for the DGCNN model.")
    parser.add_argument('--num_patches', type=int, default=128,
                        help="Number of patches per tooth in the point cloud.")
    parser.add_argument('--patch_size', type=int, default=32,
                        help="Size of each patch in the point cloud.")

    # Parse arguments
    args = parser.parse_args()
    
    # Print arguments for verification
    print("Training with the following arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    
    # Run training
    train_model(args)