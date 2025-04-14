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
    
    # Calculate d_model and check compatibility with n_head
    num_teeth = 14
    d_model = num_teeth * args.embed_dim
    if d_model % args.n_head != 0:
        raise ValueError(
            f"d_model ({d_model}) must be divisible by n_head ({args.n_head}). "
            f"Choose a different n_head or adjust embed_dim ({args.embed_dim})."
        )
    
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
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    # Initialize models
    dgcnn = DGCNN(in_channels=13, embed_dim=args.embed_dim, num_teeth=14, k=10).to(device)
    transformer = StageTransformer(
        d_model=d_model,
        max_stages=args.max_stages,
        n_head=args.n_head,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers
    ).to(device)
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
    patience_counter = 0

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_transform_loss = 0
        train_stages_loss = 0
        train_stages_loss_unnorm = 0
        train_total_loss = 0
        
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, batch in enumerate(train_loader):
            cordinates, targets, true_num_stages = batch
            cordinates, targets, true_num_stages = [
                x.to(device) for x in [cordinates, targets, true_num_stages]
            ]
            
            # Normalize targets and true_num_stages
            targets = targets / (torch.abs(targets).max() + 1e-8)
            true_num_stages_normalized = true_num_stages.float() / args.max_stages
            
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}: true_num_stages = {true_num_stages.tolist()}")
            
            with torch.amp.autocast('cuda'):
                transforms_sequence, num_stages_pred = model(
                    cordinates, teacher_forcing=targets, 
                    true_num_stages=true_num_stages,
                    epoch=epoch, total_epochs=args.epochs
                )
                
                # Compute transform loss
                transform_loss = 0
                for b in range(cordinates.size(0)):
                    n_stages = true_num_stages[b].item()
                    if n_stages > 0:
                        pred = transforms_sequence[b, :n_stages, :, :]
                        tgt = targets[b, :n_stages, :, :]
                        transform_loss += transform_criterion(pred, tgt)
                transform_loss = transform_loss / cordinates.size(0)
                
                # Compute stages loss (normalized and unnormalized)
                stages_loss = stages_criterion(
                    num_stages_pred.squeeze(-1).float() / args.max_stages,
                    true_num_stages_normalized
                )
                stages_loss_unnorm = stages_criterion(
                    num_stages_pred.squeeze(-1).float(),
                    true_num_stages.float()
                )
                
                # Combine losses
                loss = transform_loss + args.stages_loss_weight * stages_loss
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            train_transform_loss += transform_loss.item()
            train_stages_loss += stages_loss.item()
            train_stages_loss_unnorm += stages_loss_unnorm.item()
            train_total_loss += loss.item() * accumulation_steps
        
        # Log average losses per epoch
        print(f"Epoch {epoch+1}/{args.epochs}, Train Transform Loss: {train_transform_loss / len(train_loader):.4f}")
        print(f"Epoch {epoch+1}/{args.epochs}, Train Stages Loss (Normalized): {train_stages_loss / len(train_loader):.4f}")
        print(f"Epoch {epoch+1}/{args.epochs}, Train Stages Loss (Unnormalized): {train_stages_loss_unnorm / len(train_loader):.4f}")
        print(f"Epoch {epoch+1}/{args.epochs}, Train Total Loss: {train_total_loss / len(train_loader):.4f}")
        
        # Evaluation loop
        model.eval()
        test_transform_loss = 0
        test_stages_loss = 0
        test_stages_loss_unnorm = 0
        test_total_loss = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                cordinates, targets, true_num_stages = batch
                cordinates, targets, true_num_stages = [
                    x.to(device) for x in [cordinates, targets, true_num_stages]
                ]
                targets = targets / (torch.abs(targets).max() + 1e-8)
                true_num_stages_normalized = true_num_stages.float() / args.max_stages
                
                with torch.amp.autocast('cuda'):
                    transforms_sequence, num_stages_pred = model(cordinates)
                    max_stages = int(num_stages_pred.max().item())
                    max_stages = min(max(1, max_stages), args.max_stages)
                    
                    print(f"Test Batch {batch_idx+1}: true_num_stages = {true_num_stages.tolist()}")
                    
                    transform_loss = 0
                    for b in range(cordinates.size(0)):
                        n_stages = min(max_stages, true_num_stages[b].item())
                        if n_stages > 0:
                            pred = transforms_sequence[b, :n_stages, :, :]
                            tgt = targets[b, :n_stages, :, :]
                            transform_loss += transform_criterion(pred, tgt)
                    transform_loss = transform_loss / cordinates.size(0)
                    
                    stages_loss = stages_criterion(
                        num_stages_pred.squeeze(-1).float() / args.max_stages,
                        true_num_stages_normalized
                    )
                    stages_loss_unnorm = stages_criterion(
                        num_stages_pred.squeeze(-1).float(),
                        true_num_stages.float()
                    )
                    
                    loss = transform_loss + args.stages_loss_weight * stages_loss
                
                test_transform_loss += transform_loss.item()
                test_stages_loss += stages_loss.item()
                test_stages_loss_unnorm += stages_loss_unnorm.item()
                test_total_loss += loss.item()
        
        # Log average test losses
        print(f"Epoch {epoch+1}/{args.epochs}, Test Transform Loss: {test_transform_loss / len(test_loader):.4f}")
        print(f"Epoch {epoch+1}/{args.epochs}, Test Stages Loss (Normalized): {test_stages_loss / len(test_loader):.4f}")
        print(f"Epoch {epoch+1}/{args.epochs}, Test Stages Loss (Unnormalized): {test_stages_loss_unnorm / len(test_loader):.4f}")
        print(f"Epoch {epoch+1}/{args.epochs}, Test Total Loss: {test_total_loss / len(test_loader):.4f}")
        
        # Early stopping (optional)
        if args.early_stopping:
            test_loss_avg = test_total_loss / len(test_loader)
            if test_loss_avg < best_test_loss:
                best_test_loss = test_loss_avg
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        scheduler.step()

    # Save models
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(dgcnn.state_dict(), os.path.join(args.output_dir, "dgcnn.pth"))
    torch.save(model.stage_predictor.state_dict(), os.path.join(args.output_dir, "stage_predictor.pth"))
    torch.save(transformer.state_dict(), os.path.join(args.output_dir, "stage_transformer.pth"))
    torch.save(model.state_dict(), os.path.join(args.output_dir, "ortho_dgcnn.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an OrthoDGCNN model for orthodontic transformation prediction.")
    parser.add_argument('--data_dir', type=str, default="/media/osama/sm/Sample_data")
    parser.add_argument('--max_stages', type=int, default=20)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--output_dir', type=str, default="output")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--stages_loss_weight', type=float, default=1.0)  # Increased from 0.1 to 1.0
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--num_patches', type=int, default=128)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--early_stopping', action='store_true', default=False, help="Enable early stopping")
    parser.add_argument('--patience', type=int, default=10, help="Patience for early stopping")
    parser.add_argument('--n_head', type=int, default=16, help="Number of attention heads in transformer")
    parser.add_argument('--num_encoder_layers', type=int, default=6, help="Number of encoder layers in transformer")
    parser.add_argument('--num_decoder_layers', type=int, default=6, help="Number of decoder layers in transformer")

    args = parser.parse_args()
    print("Training with the following arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    train_model(args)