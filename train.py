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
import logging

# Set environment variables for CUDA
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True

# Set up logging
def setup_logging(log_file):
    logger = logging.getLogger('TrainLogger')
    logger.setLevel(logging.INFO)
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Create formatters and add them to handlers
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def train_model(args):
    # Initialize logger with the user-specified log file path
    logger = setup_logging(args.log_file)
    
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
    
    # Initialize datasets, passing the log_file parameter
    train_dataset = JawTeethDataset(
        args.data_dir, 
        max_stages=args.max_stages, 
        num_patches=args.num_patches, 
        patch_size=args.patch_size, 
        channels=13, 
        split='train', 
        train_ratio=args.train_ratio, 
        inference=False,
        log_file=args.log_file
    )
    test_dataset = JawTeethDataset(
        args.data_dir, 
        max_stages=args.max_stages, 
        num_patches=args.num_patches, 
        patch_size=args.patch_size, 
        channels=13, 
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
    
    # Initialize weights to prevent numerical instability
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
    
    # Optimizer and loss functions
    optimizer = torch.optim.Adam(
        list(dgcnn.parameters()) + list(transformer.parameters()) + 
        list(model.stage_predictor.parameters()) + list(model.transform_head.parameters()), 
        lr=args.lr,
        weight_decay=1e-4
    )
    transform_criterion = nn.MSELoss()
    stages_criterion = nn.CrossEntropyLoss()
    
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
            
            # Ensure true_num_stages is torch.long
            true_num_stages = true_num_stages.long()
            logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}: values = {true_num_stages.tolist()}")
            
            # Debug: Check for nan/inf in inputs
            if torch.isnan(cordinates).any() or torch.isinf(cordinates).any():
                logger.warning(f"Epoch {epoch+1}, Batch {batch_idx+1}: cordinates contains nan/inf")
            if torch.isnan(targets).any() or torch.isinf(targets).any():
                logger.warning(f"Epoch {epoch+1}, Batch {batch_idx+1}: targets contains nan/inf")
                # Replace nan/inf with 0 to prevent propagation
                targets = torch.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize targets safely
            max_abs_targets = torch.abs(targets).max()
            if torch.isnan(max_abs_targets) or torch.isinf(max_abs_targets) or max_abs_targets == 0:
                logger.warning(f"Epoch {epoch+1}, Batch {batch_idx+1}: Invalid max_abs_targets ({max_abs_targets}), skipping normalization")
                normalized_targets = targets
            else:
                normalized_targets = targets / (max_abs_targets + 1e-8)
            
            
            with torch.amp.autocast('cuda'):
                transforms_sequence, stage_logits = model(
                    cordinates, teacher_forcing=normalized_targets, 
                    true_num_stages=true_num_stages,
                    epoch=epoch, total_epochs=args.epochs
                )
                
                # Debug: Check for nan/inf in model outputs
                if torch.isnan(transforms_sequence).any() or torch.isinf(transforms_sequence).any():
                    logger.warning(f"Epoch {epoch+1}, Batch {batch_idx+1}: transforms_sequence contains nan/inf")
                if torch.isnan(stage_logits).any() or torch.isinf(stage_logits).any():
                    logger.warning(f"Epoch {epoch+1}, Batch {batch_idx+1}: stage_logits contains nan/inf")
                
                # Compute transform loss using true_num_stages
                transform_loss = 0
                valid_samples = 0
                for b in range(cordinates.size(0)):
                    n_stages = true_num_stages[b].item()
                    if n_stages > 0:
                        pred = transforms_sequence[b, :n_stages, :, :]
                        tgt = normalized_targets[b, :n_stages, :, :]
                        # Skip if pred or tgt contains nan/inf
                        if torch.isnan(pred).any() or torch.isinf(pred).any() or torch.isnan(tgt).any() or torch.isinf(tgt).any():
                            logger.warning(f"Epoch {epoch+1}, Batch {batch_idx+1}, Sample {b}: Skipping transform loss due to nan/inf")
                            continue
                        transform_loss += transform_criterion(pred, tgt)
                        valid_samples += 1
                
                # Handle case where all samples are skipped
                if valid_samples == 0:
                    logger.warning(f"Epoch {epoch+1}, Batch {batch_idx+1}: No valid samples for transform loss, setting to 0")
                    transform_loss = torch.tensor(0.0, device=device)
                else:
                    transform_loss = transform_loss / valid_samples
                
                # Compute stages loss (cross-entropy)
                stage_targets = true_num_stages - 1  # Shape: (batch_size,), values in [0, max_stages-1]
                stage_targets = stage_targets.long()  # Ensure type is torch.long
                stages_loss = stages_criterion(stage_logits, stage_targets)
                
                # Compute unnormalized stages loss (MSE between predicted and true stages)
                num_stages_pred = torch.argmax(stage_logits, dim=1) + 1  # Shape: (batch_size,)
                stages_loss_unnorm = ((num_stages_pred.float() - true_num_stages.float()) ** 2).mean()
                
                # Combine losses
                loss = transform_loss + args.stages_loss_weight * stages_loss
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            # Adjust gradient clipping to be more aggressive
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)  # Reduced from 0.5 to 0.1
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            train_transform_loss += transform_loss.item()
            train_stages_loss += stages_loss.item()
            train_stages_loss_unnorm += stages_loss_unnorm.item()
            train_total_loss += loss.item() * accumulation_steps
        
        # Log average losses per epoch
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Train Transform Loss (MSE): {train_transform_loss / len(train_loader):.4f}")
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Train Stages Loss (Cross-Entropy): {train_stages_loss / len(train_loader):.4f}")
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Train Stages Loss (Unnormalized MSE): {train_stages_loss_unnorm / len(train_loader):.4f}")
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Train Total Loss: {train_total_loss / len(train_loader):.4f}")
        
        # Evaluation loop
        model.eval()
        test_transform_loss = 0
        test_transform_loss_pred_stages = 0
        test_stages_loss = 0
        test_stages_loss_unnorm = 0
        test_total_loss = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                cordinates, targets, true_num_stages = batch
                cordinates, targets, true_num_stages = [
                    x.to(device) for x in [cordinates, targets, true_num_stages]
                ]
                true_num_stages = true_num_stages.long()  # Ensure type is torch.long
                
                # Debug: Check for nan/inf in test inputs
                if torch.isnan(cordinates).any() or torch.isinf(cordinates).any():
                    logger.warning(f"Test Epoch {epoch+1}, Batch {batch_idx+1}: cordinates contains nan/inf")
                if torch.isnan(targets).any() or torch.isinf(targets).any():
                    logger.warning(f"Test Epoch {epoch+1}, Batch {batch_idx+1}: targets contains nan/inf")
                    targets = torch.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Normalize targets safely
                max_abs_targets = torch.abs(targets).max()
                if torch.isnan(max_abs_targets) or torch.isinf(max_abs_targets) or max_abs_targets == 0:
                    logger.warning(f"Test Epoch {epoch+1}, Batch {batch_idx+1}: Invalid max_abs_targets ({max_abs_targets}), skipping normalization")
                    normalized_targets = targets
                else:
                    normalized_targets = targets / (max_abs_targets + 1e-8)
                
                with torch.amp.autocast('cuda'):
                    transforms_sequence, stage_logits = model(cordinates)

                    # Debug: Check for nan/inf in test outputs
                    if torch.isnan(transforms_sequence).any() or torch.isinf(transforms_sequence).any():
                        logger.warning(f"Test Epoch {epoch+1}, Batch {batch_idx+1}: transforms_sequence contains nan/inf")
                    if torch.isnan(stage_logits).any() or torch.isinf(stage_logits).any():
                        logger.warning(f"Test Epoch {epoch+1}, Batch {batch_idx+1}: stage_logits contains nan/inf")
                    
                    logger.info(f"Test Batch {batch_idx+1}: true_num_stages = {true_num_stages.tolist()}")
                    
                    # Transform loss using true_num_stages
                    transform_loss = 0
                    valid_samples = 0
                    for b in range(cordinates.size(0)):
                        n_stages = true_num_stages[b].item()
                        if n_stages > 0:
                            pred = transforms_sequence[b, :n_stages, :, :]
                            tgt = normalized_targets[b, :n_stages, :, :]
                            if torch.isnan(pred).any() or torch.isinf(pred).any() or torch.isnan(tgt).any() or torch.isinf(tgt).any():
                                logger.warning(f"Test Epoch {epoch+1}, Batch {batch_idx+1}, Sample {b}: Skipping transform loss due to nan/inf")
                                continue
                            transform_loss += transform_criterion(pred, tgt)
                            valid_samples += 1
                    
                    if valid_samples == 0:
                        logger.warning(f"Test Epoch {epoch+1}, Batch {batch_idx+1}: No valid samples for transform loss, setting to 0")
                        transform_loss = torch.tensor(0.0, device=device)
                    else:
                        transform_loss = transform_loss / valid_samples
                    
                    # Transform loss using predicted num_stages (for comparison)
                    transform_loss_pred_stages = 0
                    num_stages_pred = torch.argmax(stage_logits, dim=1) + 1
                    valid_samples_pred = 0
                    for b in range(cordinates.size(0)):
                        n_stages = true_num_stages[b].item()
                        if n_stages > 0:
                            pred = transforms_sequence[b, :n_stages, :, :]
                            tgt = normalized_targets[b, :n_stages, :, :]
                            if torch.isnan(pred).any() or torch.isinf(pred).any() or torch.isnan(tgt).any() or torch.isinf(tgt).any():
                                logger.warning(f"Test Epoch {epoch+1}, Batch {batch_idx+1}, Sample {b}: Skipping transform loss (pred stages) due to nan/inf")
                                continue
                            transform_loss_pred_stages += transform_criterion(pred, tgt)
                            valid_samples_pred += 1
                    
                    if valid_samples_pred == 0:
                        logger.warning(f"Test Epoch {epoch+1}, Batch {batch_idx+1}: No valid samples for transform loss (pred stages), setting to 0")
                        transform_loss_pred_stages = torch.tensor(0.0, device=device)
                    else:
                        transform_loss_pred_stages = transform_loss_pred_stages / valid_samples_pred
                    
                    # Stages loss (cross-entropy)
                    stage_targets = true_num_stages - 1
                    stage_targets = stage_targets.long()  # Ensure type is torch.long
                    stages_loss = stages_criterion(stage_logits, stage_targets)
                    
                    # Unnormalized stages loss (MSE)
                    stages_loss_unnorm = ((num_stages_pred.float() - true_num_stages.float()) ** 2).mean()
                    
                    # Combine losses
                    loss = transform_loss + args.stages_loss_weight * stages_loss
                
                test_transform_loss += transform_loss.item()
                test_transform_loss_pred_stages += transform_loss_pred_stages.item()
                test_stages_loss += stages_loss.item()
                test_stages_loss_unnorm += stages_loss_unnorm.item()
                test_total_loss += loss.item()
        
        # Log average test losses
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Test Transform Loss (True Stages) (MSE): {test_transform_loss / len(test_loader):.4f}")
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Test Transform Loss (Pred Stages) (MSE): {test_transform_loss_pred_stages / len(test_loader):.4f}")
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Test Stages Loss (Cross-Entropy): {test_stages_loss / len(test_loader):.4f}")
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Test Stages Loss (Unnormalized MSE): {test_stages_loss_unnorm / len(test_loader):.4f}")
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Test Total Loss: {test_total_loss / len(test_loader):.4f}")
        
        # Early stopping (optional)
        if args.early_stopping:
            test_loss_avg = test_total_loss / len(test_loader)
            if test_loss_avg < best_test_loss:
                best_test_loss = test_loss_avg
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        scheduler.step()

    # Save models
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(dgcnn.state_dict(), os.path.join(args.output_dir, "dgcnn.pth"))
    torch.save(model.stage_predictor.state_dict(), os.path.join(args.output_dir, "stage_predictor.pth"))
    torch.save(transformer.state_dict(), os.path.join(args.output_dir, "stage_transformer.pth"))
    torch.save(model.state_dict(), os.path.join(args.output_dir, "ortho_dgcnn.pth"))
    logger.info("Models saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an OrthoDGCNN model for orthodontic transformation prediction.")
    parser.add_argument('--data_dir', type=str, default="/media/osama/sm/Sample_data")
    parser.add_argument('--max_stages', type=int, default=20)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--output_dir', type=str, default="output")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--stages_loss_weight', type=float, default=10.0)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--num_patches', type=int, default=128)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--early_stopping', action='store_true', default=False, help="Enable early stopping")
    parser.add_argument('--patience', type=int, default=10, help="Patience for early stopping")
    parser.add_argument('--n_head', type=int, default=16, help="Number of attention heads in transformer")
    parser.add_argument('--num_encoder_layers', type=int, default=6, help="Number of encoder layers in transformer")
    parser.add_argument('--num_decoder_layers', type=int, default=6, help="Number of decoder layers in transformer")
    parser.add_argument('--log_file', type=str, default="training_log.txt", help="Path to the log file")

    args = parser.parse_args()
    
    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(args.log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Set up logging for the main script
    logger = setup_logging(args.log_file)
    logger.info("Training with the following arguments:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    
    train_model(args)