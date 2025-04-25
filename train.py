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
from torch.cuda.amp import GradScaler, autocast
import pandas as pd

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
    def __init__(self, beta=0.5, alpha=10.0, gamma=0.05, epsilon=1e-6):
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    
    def forward(self, pred, target, activity_mask=None, stage_weights=None):
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

def consistency_loss(pred_transforms, cumulative_transforms, true_num_stages, max_stages, device):
    batch_size = pred_transforms.size(0)
    loss = 0.0
    for i in range(batch_size):
        num_stages = min(true_num_stages[i].item(), max_stages)
        pred_sum = pred_transforms[i, :num_stages].sum(dim=0)  # [num_teeth, 6]
        loss += nn.L1Loss()(pred_sum, cumulative_transforms[i])
    return loss / batch_size

def main(args):
    logger = setup_logging(args.log_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize datasets
    train_dataset = JawTeethDataset(
        data_dir=args.data_dir,
        max_stages=args.max_stages,
        split='train',
        log_file=args.log_file
    )
    val_dataset = JawTeethDataset(
        data_dir=args.data_dir,
        max_stages=args.max_stages,
        split='val',
        log_file=args.log_file
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize models
    dgcnn = DGCNN(k=args.dgcnn_k, embed_dim=args.embed_dim).to(device)
    mvit = MViTv2(
        embed_dim=args.embed_dim,
        num_teeth=args.num_teeth,
        max_stages=args.max_stages,
        depths=args.mvit_depths,
        num_heads=args.mvit_num_heads,
        mlp_ratio=args.mvit_mlp_ratio,
        drop_path_rate=args.mvit_drop_path_rate,
        decoder_layers=args.mvit_decoder_layers,
        teacher_forcing=args.teacher_forcing_prob > 0,
    ).to(device)
    model = OrthoDGCNNModel(
        dgcnn=dgcnn,
        mvit=mvit,
        max_stages=args.max_stages,
        num_teeth=args.num_teeth,
        embed_dim=args.embed_dim,
        teacher_forcing=args.teacher_forcing_prob > 0,
        depths=args.mvit_depths,
        num_heads=args.mvit_num_heads,
        mlp_ratio=args.mvit_mlp_ratio,
        drop_path_rate=args.mvit_drop_path_rate
    ).to(device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Loss functions
    smooth_l1_loss = WeightedSmoothL1Loss().to(device)
    mae_loss = nn.L1Loss().to(device)
    bce_loss = nn.BCEWithLogitsLoss().to(device)
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            feats, transformations, cumulative_transforms, num_stages, activity, param_activity = [x.to(device) for x in batch]
            
            optimizer.zero_grad()
            with autocast():
                pred_transforms, activity_logits, param_activity_logits, pred_cumulative = model(
                    feats,
                    targets=transformations,
                    cumulative_targets=cumulative_transforms,
                    epoch=epoch,
                    total_epochs=args.epochs
                )
                
                # Compute stage weights
                batch_size = feats.size(0)
                stage_weights = torch.ones(batch_size, args.max_stages, device=device)
                for i in range(batch_size):
                    stage_weights[i, num_stages[i]:] = 0.0
                
                # Losses
                loss_transform = smooth_l1_loss(pred_transforms, transformations, activity, stage_weights)
                loss_cumulative = mae_loss(pred_cumulative, cumulative_transforms)
                loss_consistency = consistency_loss(pred_transforms, cumulative_transforms, num_stages, args.max_stages, device)
                loss_zero = zero_prediction_loss(pred_transforms, transformations, activity, stage_weights)
                loss_activity = bce_loss(activity_logits, activity)
                loss_param_activity = bce_loss(param_activity_logits, param_activity)
                
                total_loss = (
                    args.w_transform * loss_transform +
                    args.w_cumulative * loss_cumulative +
                    args.w_consistency * loss_consistency +
                    args.w_zero * loss_zero +
                    args.w_activity * loss_activity +
                    args.w_param_activity * loss_param_activity
                )
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += total_loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}/{len(train_loader)}, "
                            f"Loss: {total_loss.item():.4f}, Transform: {loss_transform.item():.4f}, "
                            f"Cumulative: {loss_cumulative.item():.4f}, Consistency: {loss_consistency.item():.4f}, "
                            f"Zero: {loss_zero.item():.4f}, Activity: {loss_activity.item():.4f}, "
                            f"Param Activity: {loss_param_activity.item():.4f}")
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                feats, transformations, cumulative_transforms, num_stages, activity, param_activity = [x.to(device) for x in batch]
                
                with autocast():
                    pred_transforms, activity_logits, param_activity_logits, pred_cumulative = model(
                        feats,
                        targets=transformations,
                        cumulative_targets=cumulative_transforms,
                        epoch=epoch,
                        total_epochs=args.epochs
                    )
                    
                    stage_weights = torch.ones(feats.size(0), args.max_stages, device=device)
                    for i in range(feats.size(0)):
                        stage_weights[i, num_stages[i]:] = 0.0
                    
                    loss_transform = smooth_l1_loss(pred_transforms, transformations, activity, stage_weights)
                    loss_cumulative = mae_loss(pred_cumulative, cumulative_transforms)
                    loss_consistency = consistency_loss(pred_transforms, cumulative_transforms, num_stages, args.max_stages, device)
                    loss_zero = zero_prediction_loss(pred_transforms, transformations, activity, stage_weights)
                    loss_activity = bce_loss(activity_logits, activity)
                    loss_param_activity = bce_loss(param_activity_logits, param_activity)
                    
                    total_loss = (
                        args.w_transform * loss_transform +
                        args.w_cumulative * loss_cumulative +
                        args.w_consistency * loss_consistency +
                        args.w_zero * loss_zero +
                        args.w_activity * loss_activity +
                        args.w_param_activity * loss_param_activity
                    )
                
                val_loss += total_loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step()
        
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pth'))
            logger.info(f"Saved best model at epoch {epoch+1} with val_loss {val_loss:.4f}")
    
    logger.info("Training completed")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./output', help='Path to save checkpoints')
    parser.add_argument('--log_file', type=str, default='training_log.txt', help='Path to log file')
    
    # DGCNN hyperparameters
    parser.add_argument('--dgcnn_k', type=int, default=20, help='Number of neighbors for DGCNN')
    parser.add_argument('--embed_dim', type=int, default=96, help='Embedding dimension')
    
    # CumulativeTransformationModel hyperparameters
    parser.add_argument('--cumulative_num_heads', type=int, default=4, help='Number of attention heads in CumulativeTransformationModel')
    
    # MViTv2 hyperparameters
    parser.add_argument('--mvit_depths', type=lambda s: ast.literal_eval(s), default="[1, 2, 11, 2]", help='Depths of MViTv2 stages')
    parser.add_argument('--mvit_num_heads', type=lambda s: ast.literal_eval(s), default="[3, 3, 3, 3]", help='Number of heads per stage')
    parser.add_argument('--mvit_mlp_ratio', type=float, default=4.0, help='MLP ratio in MViTv2')
    parser.add_argument('--mvit_drop_path_rate', type=float, default=0.1, help='Drop path rate in MViTv2')
    parser.add_argument('--mvit_decoder_layers', type=int, default=1, help='Number of decoder layers')
    parser.add_argument('--mvit_pretrained_model', type=str, default='mvitv2_small', help='Pretrained MViTv2 model name')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--teacher_forcing_prob', type=float, default=0.5, help='Teacher forcing probability')
    parser.add_argument('--cumulative_teacher_forcing_prob', type=float, default=0.5, help='Cumulative teacher forcing probability')
    parser.add_argument('--max_stages', type=int, default=25, help='Maximum number of stages')
    parser.add_argument('--num_teeth', type=int, default=14, help='Number of teeth')
    
    # Loss weights
    parser.add_argument('--w_transform', type=float, default=1.0, help='Weight for transform loss')
    parser.add_argument('--w_cumulative', type=float, default=1.0, help='Weight for cumulative loss')
    parser.add_argument('--w_consistency', type=float, default=0.5, help='Weight for consistency loss')
    parser.add_argument('--w_zero', type=float, default=0.2, help='Weight for zero prediction loss')
    parser.add_argument('--w_activity', type=float, default=0.5, help='Weight for activity loss')
    parser.add_argument('--w_param_activity', type=float, default=0.5, help='Weight for param activity loss')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)