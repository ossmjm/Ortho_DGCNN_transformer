import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import argparse
import logging
import ast
from dataset import JawTeethDataset
from models.DGCNN import DGCNN
from models.MViT import MViTv2
from models.OrthoDGCNN import OrthoDGCNNModel
from torch.optim.lr_scheduler import CosineAnnealingLR

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

class WeightedSmoothL1Loss(nn.Module):
    def __init__(self, beta=0.5, alpha=10.0, gamma=0.05, epsilon=1e-6, max_value=100.0):
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_value = max_value
    
    def forward(self, pred, target, activity_mask=None, stage_weights=None):
        pred = torch.clamp(pred, -self.max_value, self.max_value)
        target = torch.clamp(target, -self.max_value, self.max_value)
        diff = torch.abs(pred - target)
        smooth_l1 = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        weights = torch.exp(-self.alpha * torch.clamp(diff, min=self.epsilon)) + self.gamma
        if activity_mask is not None:
            smooth_l1 = smooth_l1 * activity_mask
            weights = weights * activity_mask
        if stage_weights is not None:
            stage_weights_expanded = stage_weights.unsqueeze(-1).unsqueeze(-1)
            smooth_l1 = smooth_l1 * stage_weights_expanded
            weights = weights * stage_weights_expanded
        num_active = (activity_mask * stage_weights_expanded).sum() if activity_mask is not None and stage_weights is not None else smooth_l1.numel()
        num_active = num_active.clamp(min=self.epsilon)
        loss = (weights * smooth_l1).sum() / num_active
        return loss

def zero_prediction_loss(pred, target, activity_mask, stage_weights, threshold=0.1):
    zero_mask = (target == 0).float() * activity_mask
    non_zero_pred = torch.abs(pred) * zero_mask
    loss = torch.relu(non_zero_pred - threshold) ** 2
    stage_weights_expanded = stage_weights.unsqueeze(-1).unsqueeze(-1)
    num_active = (zero_mask * stage_weights_expanded).sum().clamp(min=1e-6)
    return (loss * stage_weights_expanded).sum() / num_active

def compute_loss(transforms_sequence, activity_logits, param_activity_logits, targets, activity_labels, param_activity_labels, true_num_stages, max_stages, device, logger):
    trans_loss_fn = WeightedSmoothL1Loss(beta=0.5, alpha=5.0, gamma=0.1)
    rot_loss_fn = WeightedSmoothL1Loss(beta=0.5, alpha=10.0, gamma=0.05)
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    
    pred_trans = transforms_sequence[:, :, :, :3]
    pred_rot = transforms_sequence[:, :, :, 3:]
    target_trans = targets[:, :, :, :3]
    target_rot = targets[:, :, :, 3:]
    trans_activity = param_activity_labels[:, :, :, :3]
    rot_activity = param_activity_labels[:, :, :, 3:]
    
    batch_size = transforms_sequence.size(0)
    stage_weights = torch.zeros(batch_size, max_stages, device=device)
    for b in range(batch_size):
        stage_weights[b, :true_num_stages[b]] = 1.0
    
    loss_trans = trans_loss_fn(pred_trans, target_trans, trans_activity, stage_weights)
    loss_rot = rot_loss_fn(pred_rot, target_rot, rot_activity, stage_weights)
    zero_trans_loss = zero_prediction_loss(pred_trans, target_trans, trans_activity, stage_weights)
    zero_rot_loss = zero_prediction_loss(pred_rot, target_rot, rot_activity, stage_weights)
    
    activity_loss = bce_loss_fn(activity_logits, activity_labels)
    param_activity_loss = bce_loss_fn(param_activity_logits, param_activity_labels)
    
    padded_loss = 0.0
    for b in range(batch_size):
        true_stages = true_num_stages[b].item()
        if true_stages < max_stages:
            padded_loss += torch.mean(transforms_sequence[b, true_stages:, :, :]**2)
    padded_loss = padded_loss / batch_size if batch_size > 0 else 0.0
    sparsity_loss = torch.mean((transforms_sequence * (1 - param_activity_labels))**2)
    
    tooth_errors = torch.mean(torch.abs(transforms_sequence - targets) * activity_labels.unsqueeze(-1), dim=(0, 1, 3))
    for tooth_idx, fdi in enumerate([31, 32, 33, 34, 35, 36, 37, 41, 42, 43, 44, 45, 46, 47]):
        logger.debug(f"Tooth {fdi}: Mean Absolute Error = {tooth_errors[tooth_idx]:.4f}")
    
    alpha, beta, gamma, delta, epsilon, zeta = 10.0, 50.0, 0.1, 30.0, 5.0, 5.0
    total_loss = (alpha * loss_trans + beta * loss_rot + gamma * padded_loss + 
                  delta * sparsity_loss + epsilon * activity_loss + zeta * param_activity_loss + 
                  zero_trans_loss + zero_rot_loss)
    
    return (total_loss, loss_trans, rot_loss, padded_loss, sparsity_loss, 
            activity_loss, param_activity_loss, zero_trans_loss, zero_rot_loss)

def parse_args():
    parser = argparse.ArgumentParser(description="Train OrthoDGCNN Model")
    parser.add_argument('--data-dir', type=str, default='./data', help='Path to dataset directory')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max-stages', type=int, default=25, help='Maximum number of stages')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Ratio of data for training')
    parser.add_argument('--log-file', type=str, default='training_log.txt', help='Log file path')
    parser.add_argument('--teacher-forcing', action='store_true', help='Enable teacher forcing')
    parser.add_argument('--embed-dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--depths', type=str, default='[1, 2, 11, 2]', help='Number of blocks per stage')
    parser.add_argument('--num-heads', type=str, default='[4, 4, 8, 8]', help='Number of attention heads per stage')
    parser.add_argument('--mlp-ratio', type=float, default=4.0, help='MLP expansion ratio')
    parser.add_argument('--drop-path-rate', type=float, default=0.2, help='Drop path rate')
    return parser.parse_args()

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
    depths = ast.literal_eval(args.depths)
    num_heads = ast.literal_eval(args.num_heads)
    
    train_dataset = JawTeethDataset(
        args.data_dir, 
        max_stages=args.max_stages, 
        split='train', 
        train_ratio=args.train_ratio, 
        inference=False,
        log_file=args.log_file,
    )
    test_dataset = JawTeethDataset(
        args.data_dir, 
        max_stages=args.max_stages, 
        split='test', 
        train_ratio=args.train_ratio, 
        inference=False,
        log_file=args.log_file,
    )
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    logger.info(f"Number of training batches: {len(train_loader)}")
    logger.info(f"Number of test batches: {len(test_loader)}")
    
    dgcnn = DGCNN(in_channels=13, embed_dim=args.embed_dim, num_teeth=14, k=10).to(device)
    mvit = MViTv2(
        embed_dim=args.embed_dim,
        num_teeth=14,
        max_stages=args.max_stages,
        depths=depths,
        num_heads=num_heads,
        mlp_ratio=args.mlp_ratio,
        drop_path_rate=args.drop_path_rate,
        teacher_forcing=args.teacher_forcing
    ).to(device)
    model = OrthoDGCNNModel(
        dgcnn, 
        mvit, 
        max_stages=args.max_stages,
        num_teeth=14,
        embed_dim=args.embed_dim,
        teacher_forcing=args.teacher_forcing,
        depths=depths,
        num_heads=num_heads,
        mlp_ratio=args.mlp_ratio,
        drop_path_rate=args.drop_path_rate
    ).to(device)
    
    def initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Parameter):
            nn.init.uniform_(module, -0.1, 0.1)
    
    model.apply(initialize_weights)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr,
        weight_decay=1e-3
    )
    scaler = torch.amp.GradScaler('cuda')
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    accumulation_steps = 2
    best_test_loss = float('inf')
    patience_counter = 0
    patience = 10
    model_save_path = 'best_model.pth'

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_trans_loss = 0
        train_rot_loss = 0
        train_padded_loss = 0
        train_sparsity_loss = 0
        train_activity_loss = 0
        train_param_activity_loss = 0
        train_zero_trans_loss = 0
        train_zero_rot_loss = 0
        optimizer.zero_grad(set_to_none=True)
        
        tf_prob = max(0.0, 1.0 - epoch / (args.epochs * 0.75)) if args.teacher_forcing else 0.0
        
        for batch_idx, (cordinates, targets, true_num_stages, activity_labels, param_activity_labels) in enumerate(train_loader):
            cordinates, targets, true_num_stages, activity_labels, param_activity_labels = (
                cordinates.to(device), targets.to(device), true_num_stages.to(device), 
                activity_labels.to(device), param_activity_labels.to(device)
            )
            
            if torch.isnan(cordinates).any() or torch.isinf(cordinates).any():
                logger.warning(f"Epoch {epoch+1}, Batch {batch_idx+1}: cordinates contains nan/inf")
            if torch.isnan(targets).any() or torch.isinf(targets).any():
                logger.warning(f"Epoch {epoch+1}, Batch {batch_idx+1}: targets contains nan/inf")
                targets = torch.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
            if torch.isnan(activity_labels).any() or torch.isinf(activity_labels).any():
                logger.warning(f"Epoch {epoch+1}, Batch {batch_idx+1}: activity_labels contains nan/inf")
                activity_labels = torch.nan_to_num(activity_labels, nan=0.0, posinf=0.0, neginf=0.0)
            if torch.isnan(param_activity_labels).any() or torch.isinf(param_activity_labels).any():
                logger.warning(f"Epoch {epoch+1}, Batch {batch_idx+1}: param_activity_labels contains nan/inf")
                param_activity_labels = torch.nan_to_num(param_activity_labels, nan=0.0, posinf=0.0, neginf=0.0)
            
            with torch.amp.autocast('cuda'):
                transforms_sequence, activity_logits, param_activity_logits = model(
                    cordinates, 
                    targets=targets if torch.rand(1).item() < tf_prob else None,
                    epoch=epoch, 
                    total_epochs=args.epochs
                )
                
                if torch.isnan(transforms_sequence).any() or torch.isinf(transforms_sequence).any():
                    logger.warning(f"Epoch {epoch+1}, Batch {batch_idx+1}: transforms_sequence contains nan/inf")
                    continue
                
                loss, trans_loss, rot_loss, padded_loss, sparsity_loss, activity_loss, param_activity_loss, zero_trans_loss, zero_rot_loss = compute_loss(
                    transforms_sequence, activity_logits, param_activity_logits, targets, activity_labels, 
                    param_activity_labels, true_num_stages, args.max_stages, device, logger
                )
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            train_loss += loss.item() * accumulation_steps
            train_trans_loss += trans_loss.item()
            train_rot_loss += rot_loss.item()
            train_padded_loss += padded_loss.item()
            train_sparsity_loss += sparsity_loss.item()
            train_activity_loss += activity_loss.item()
            train_param_activity_loss += param_activity_loss.item()
            train_zero_trans_loss += zero_trans_loss.item()
            train_zero_rot_loss += zero_rot_loss.item()
            
            logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}: Total Loss = {loss.item() * accumulation_steps:.6f}, "
                       f"Translation Loss = {trans_loss.item():.6f}, Rotation Loss = {rot_loss.item():.6f}, "
                       f"Activity Loss = {activity_loss.item():.6f}, Param Activity Loss = {param_activity_loss.item():.6f}, "
                       f"Zero Translation Loss = {zero_trans_loss.item():.6f}, Zero Rotation Loss = {zero_rot_loss.item():.6f}, "
                       f"Padded Loss = {padded_loss.item():.6f}, Sparsity Loss = {sparsity_loss.item():.6f}, "
                       f"TF Prob = {tf_prob:.2f}")
        
        avg_train_loss = train_loss / len(train_loader)
        avg_trans_loss = train_trans_loss / len(train_loader)
        avg_rot_loss = train_rot_loss / len(train_loader)
        avg_padded_loss = train_padded_loss / len(train_loader)
        avg_sparsity_loss = train_sparsity_loss / len(train_loader)
        avg_activity_loss = train_activity_loss / len(train_loader)
        avg_param_activity_loss = train_param_activity_loss / len(train_loader)
        avg_zero_trans_loss = train_zero_trans_loss / len(train_loader)
        avg_zero_rot_loss = train_zero_rot_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Train Total Loss: {avg_train_loss:.4f}, "
                   f"Train Translation Loss: {avg_trans_loss:.4f}, Train Rotation Loss: {avg_rot_loss:.4f}, "
                   f"Train Activity Loss: {avg_activity_loss:.4f}, Train Param Activity Loss: {avg_param_activity_loss:.4f}, "
                   f"Train Zero Translation Loss: {avg_zero_trans_loss:.4f}, Train Zero Rotation Loss: {avg_zero_rot_loss:.4f}, "
                   f"Train Padded Loss: {avg_padded_loss:.4f}, Train Sparsity Loss: {avg_sparsity_loss:.4f}")
        logger.info('---')
        
        model.eval()
        test_loss = 0
        test_trans_loss = 0
        test_rot_loss = 0
        test_padded_loss = 0
        test_sparsity_loss = 0
        test_activity_loss = 0
        test_param_activity_loss = 0
        test_zero_trans_loss = 0
        test_zero_rot_loss = 0
        with torch.no_grad():
            for batch_idx, (cordinates, targets, true_num_stages, activity_labels, param_activity_labels) in enumerate(test_loader):
                cordinates, targets, true_num_stages, activity_labels, param_activity_labels = (
                    cordinates.to(device), targets.to(device), true_num_stages.to(device), 
                    activity_labels.to(device), param_activity_labels.to(device)
                )
                
                if torch.isnan(cordinates).any() or torch.isinf(cordinates).any():
                    logger.warning(f"Test Epoch {epoch+1}, Batch {batch_idx+1}: cordinates contains nan/inf")
                if torch.isnan(targets).any() or torch.isinf(targets).any():
                    logger.warning(f"Test Epoch {epoch+1}, Batch {batch_idx+1}: targets contains nan/inf")
                    targets = torch.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
                if torch.isnan(activity_labels).any() or torch.isinf(activity_labels).any():
                    logger.warning(f"Test Epoch {epoch+1}, Batch {batch_idx+1}: activity_labels contains nan/inf")
                    activity_labels = torch.nan_to_num(activity_labels, nan=0.0, posinf=0.0, neginf=0.0)
                if torch.isnan(param_activity_labels).any() or torch.isinf(param_activity_labels).any():
                    logger.warning(f"Test Epoch {epoch+1}, Batch {batch_idx+1}: param_activity_labels contains nan/inf")
                    param_activity_labels = torch.nan_to_num(param_activity_labels, nan=0.0, posinf=0.0, neginf=0.0)
                
                with torch.amp.autocast('cuda'):
                    transforms_sequence, activity_logits, param_activity_logits = model(cordinates)
                    
                    if torch.isnan(transforms_sequence).any() or torch.isinf(transforms_sequence).any():
                        logger.warning(f"Test Epoch {epoch+1}, Batch {batch_idx+1}: transforms_sequence contains nan/inf")
                        continue
                    
                    loss, trans_loss, rot_loss, padded_loss, sparsity_loss, activity_loss, param_activity_loss, zero_trans_loss, zero_rot_loss = compute_loss(
                        transforms_sequence, activity_logits, param_activity_logits, targets, activity_labels, 
                        param_activity_labels, true_num_stages, args.max_stages, device, logger
                    )
                
                test_loss += loss.item()
                test_trans_loss += trans_loss.item()
                test_rot_loss += rot_loss.item()
                test_padded_loss += padded_loss.item()
                test_sparsity_loss += sparsity_loss.item()
                test_activity_loss += activity_loss.item()
                test_param_activity_loss += param_activity_loss.item()
                test_zero_trans_loss += zero_trans_loss.item()
                test_zero_rot_loss += zero_rot_loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        avg_test_trans_loss = test_trans_loss / len(test_loader)
        avg_test_rot_loss = test_rot_loss / len(test_loader)
        avg_test_padded_loss = test_padded_loss / len(test_loader)
        avg_test_sparsity_loss = test_sparsity_loss / len(test_loader)
        avg_test_activity_loss = test_activity_loss / len(test_loader)
        avg_test_param_activity_loss = test_param_activity_loss / len(test_loader)
        avg_test_zero_trans_loss = test_zero_trans_loss / len(test_loader)
        avg_test_zero_rot_loss = test_zero_rot_loss / len(test_loader)
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Test Total Loss: {avg_test_loss:.4f}, "
                   f"Test Translation Loss: {avg_test_trans_loss:.4f}, Test Rotation Loss: {avg_test_rot_loss:.4f}, "
                   f"Test Activity Loss: {avg_test_activity_loss:.4f}, Test Param Activity Loss: {avg_test_param_activity_loss:.4f}, "
                   f"Test Zero Translation Loss: {avg_test_zero_trans_loss:.4f}, Test Zero Rotation Loss: {avg_test_zero_rot_loss:.4f}, "
                   f"Test Padded Loss: {avg_test_padded_loss:.4f}, Test Sparsity Loss: {avg_test_sparsity_loss:.4f}")
        logger.info('---')
        
        scheduler.step()
        logger.info(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Saved best model at epoch {epoch+1} with test loss {best_test_loss:.4f}")
        else:
            patience_counter += 1
            logger.info(f"Patience counter: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    logger.info(f"Training completed. Best test loss: {best_test_loss:.4f}")

if __name__ == "__main__":
    args = parse_args()
    train_model(args)