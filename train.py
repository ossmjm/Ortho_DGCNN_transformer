import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import argparse
import logging
from dataset import JawTeethDataset
from models.DGCNN import DGCNN
from models.StageTransformer import StageTransformer
from models.OrthoDGCNN import OrthoDGCNNModel

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True

def setup_logging(log_file):
    logger = logging.getLogger('TrainLogger')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def compute_loss(transforms_sequence, targets, true_num_stages, max_stages, device):
    criterion = nn.MSELoss(reduction='none')
    
    # Linear stage weights: 1.0 for stage 0, decreasing to 0.0 for stage max_stages
    stage_weights = 1.0 - torch.arange(max_stages, device=device, dtype=torch.float32) / max_stages
    stage_weights = stage_weights.view(1, max_stages, 1, 1)
    
    # Separate MSE for translations (first 3 dims) and rotations (last 3 dims)
    mse_loss_trans = criterion(transforms_sequence[:, :, :, :3], targets[:, :, :, :3])
    mse_loss_rot = criterion(transforms_sequence[:, :, :, 3:], targets[:, :, :, 3:])
    
    # Apply stage weights and compute mean per batch
    mse_loss_trans = (mse_loss_trans * stage_weights).mean()
    mse_loss_rot = (mse_loss_rot * stage_weights).mean()
    
    # Padded stage regularization
    padded_loss = 0.0
    batch_size = transforms_sequence.size(0)
    for b in range(batch_size):
        true_stages = true_num_stages[b].item()
        if true_stages < max_stages:
            padded_loss += torch.mean(transforms_sequence[b, true_stages:, :, :]**2)
    padded_loss = padded_loss / batch_size if batch_size > 0 else 0.0
    
    # Combine losses: weight translations 10x more than rotations
    alpha, beta, gamma = 10.0, 1.0, 0.1
    total_loss = alpha * mse_loss_trans + beta * mse_loss_rot + gamma * padded_loss
    
    return total_loss, mse_loss_trans, mse_loss_rot, padded_loss

def train_model(args):
    logger = setup_logging(args.log_file)
    logger.info("Training with the following arguments:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    torch.cuda.empty_cache()
    logger.info(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    
    num_teeth = 14
    d_model = num_teeth * args.embed_dim
    if d_model % args.n_head != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by n_head ({args.n_head}).")
    
    train_dataset = JawTeethDataset(
        args.data_dir, 
        max_stages=args.max_stages, 
        split='train', 
        train_ratio=args.train_ratio, 
        inference=False,
        log_file=args.log_file
    )
    test_dataset = JawTeethDataset(
        args.data_dir, 
        max_stages=args.max_stages, 
        split='test', 
        train_ratio=args.train_ratio, 
        inference=False,
        log_file=args.log_file
    )
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    logger.info(f"Number of training batches: {len(train_loader)}")
    logger.info(f"Number of test batches: {len(test_loader)}")
    
    dgcnn = DGCNN(in_channels=13, embed_dim=args.embed_dim, num_teeth=14, k=10).to(device)
    transformer = StageTransformer(
        d_model=d_model,
        max_stages=args.max_stages,
        n_head=args.n_head,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers
    ).to(device)
    model = OrthoDGCNNModel(
        dgcnn, 
        transformer, 
        max_stages=args.max_stages, 
        num_teeth=14, 
        embed_dim=args.embed_dim, 
        teacher_forcing=args.teacher_forcing
    ).to(device)
    
    def initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.uniform_(module.weight, -0.1, 0.1)
    
    dgcnn.apply(initialize_weights)
    transformer.apply(initialize_weights)
    model.apply(initialize_weights)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr,
        weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    accumulation_steps = 2
    best_test_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_trans_loss = 0
        train_rot_loss = 0
        train_padded_loss = 0
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (cordinates, targets, true_num_stages) in enumerate(train_loader):
            cordinates, targets, true_num_stages = cordinates.to(device), targets.to(device), true_num_stages.to(device)
            
            if torch.isnan(cordinates).any() or torch.isinf(cordinates).any():
                logger.warning(f"Epoch {epoch+1}, Batch {batch_idx+1}: cordinates contains nan/inf")
            if torch.isnan(targets).any() or torch.isinf(targets).any():
                logger.warning(f"Epoch {epoch+1}, Batch {batch_idx+1}: targets contains nan/inf")
                targets = torch.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
            
            with torch.amp.autocast('cuda'):
                transforms_sequence = model(
                    cordinates, 
                    targets=targets,  # Pass raw targets
                    epoch=epoch, 
                    total_epochs=args.epochs
                )
                
                if torch.isnan(transforms_sequence).any() or torch.isinf(transforms_sequence).any():
                    logger.warning(f"Epoch {epoch+1}, Batch {batch_idx+1}: transforms_sequence contains nan/inf")
                    continue
                
                # Compute custom loss
                loss, mse_loss_trans, mse_loss_rot, padded_loss = compute_loss(
                    transforms_sequence, targets, true_num_stages, args.max_stages, device
                )
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            train_loss += loss.item() * accumulation_steps
            train_trans_loss += mse_loss_trans.item()
            train_rot_loss += mse_loss_rot.item()
            train_padded_loss += padded_loss.item()
            
            logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}: Total Loss = {loss.item() * accumulation_steps:.6f}, "
                       f"Translation Loss = {mse_loss_trans.item():.6f}, Rotation Loss = {mse_loss_rot.item():.6f}, "
                       f"Padded Loss = {padded_loss.item():.6f}")
        
        avg_train_loss = train_loss / len(train_loader)
        avg_trans_loss = train_trans_loss / len(train_loader)
        avg_rot_loss = train_rot_loss / len(train_loader)
        avg_padded_loss = train_padded_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Train Total Loss: {avg_train_loss:.4f}, "
                   f"Train Translation Loss: {avg_trans_loss:.4f}, Train Rotation Loss: {avg_rot_loss:.4f}, "
                   f"Train Padded Loss: {avg_padded_loss:.4f}")
        
        model.eval()
        test_loss = 0
        test_trans_loss = 0
        test_rot_loss = 0
        test_padded_loss = 0
        with torch.no_grad():
            for batch_idx, (cordinates, targets, true_num_stages) in enumerate(test_loader):
                cordinates, targets, true_num_stages = cordinates.to(device), targets.to(device), true_num_stages.to(device)
                
                if torch.isnan(cordinates).any() or torch.isinf(cordinates).any():
                    logger.warning(f"Test Epoch {epoch+1}, Batch {batch_idx+1}: cordinates contains nan/inf")
                if torch.isnan(targets).any() or torch.isinf(targets).any():
                    logger.warning(f"Test Epoch {epoch+1}, Batch {batch_idx+1}: targets contains nan/inf")
                    targets = torch.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
                
                with torch.amp.autocast('cuda'):
                    transforms_sequence = model(cordinates)
                    
                    if torch.isnan(transforms_sequence).any() or torch.isinf(transforms_sequence).any():
                        logger.warning(f"Test Epoch {epoch+1}, Batch {batch_idx+1}: transforms_sequence contains nan/inf")
                        continue
                    
                    # Compute custom loss
                    loss, mse_loss_trans, mse_loss_rot, padded_loss = compute_loss(
                        transforms_sequence, targets, true_num_stages, args.max_stages, device
                    )
                
                test_loss += loss.item()
                test_trans_loss += mse_loss_trans.item()
                test_rot_loss += mse_loss_rot.item()
                test_padded_loss += padded_loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        avg_test_trans_loss = test_trans_loss / len(test_loader)
        avg_test_rot_loss = test_rot_loss / len(test_loader)
        avg_test_padded_loss = test_padded_loss / len(test_loader)
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Test Total Loss: {avg_test_loss:.4f}, "
                   f"Test Translation Loss: {avg_test_trans_loss:.4f}, Test Rotation Loss: {avg_test_rot_loss:.4f}, "
                   f"Test Padded Loss: {avg_test_padded_loss:.4f}")
        
        if args.early_stopping:
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                patience_counter = 0
                os.makedirs(args.output_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(args.output_dir, "ortho_dgcnn_best.pth"))
                logger.info(f"Saved best model to {args.output_dir}/ortho_dgcnn_best.pth")
            else:
                patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        scheduler.step()
    
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(dgcnn.state_dict(), os.path.join(args.output_dir, "dgcnn.pth"))
    torch.save(transformer.state_dict(), os.path.join(args.output_dir, "stage_transformer.pth"))
    torch.save(model.state_dict(), os.path.join(args.output_dir, "ortho_dgcnn.pth"))
    logger.info("Models saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an OrthoDGCNN model for orthodontic transformation prediction.")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--max_stages', type=int, default=25)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--output_dir', type=str, default="output")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--early_stopping', action='store_true', default=False)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--n_head', type=int, default=16)
    parser.add_argument('--num_encoder_layers', type=int, default=6)
    parser.add_argument('--num_decoder_layers', type=int, default=6)
    parser.add_argument('--log_file', type=str, default="training_log.txt")
    parser.add_argument('--teacher_forcing', action='store_true', default=False, help="Use teacher forcing during training")
    
    args = parser.parse_args()
    
    log_dir = os.path.dirname(args.log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    train_model(args)