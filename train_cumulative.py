import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import argparse
from dataset import CumulativeJawTeethDataset
from models.OrthoDGCNN import OrthoDGCNNModel
from losses_cumulative import CumulativeLoss, CumulativeZeroLoss, CumulativeSparsityLoss, CumulativeActivityLoss, CumulativeParamActivityLoss

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

def compute_loss(pred_cumulative, cumulative_activity_logits, cumulative_param_activity_logits,
                 cumulative_transforms, cumulative_activity, cumulative_param_activity,
                 device, logger, args):
    cumulative_loss_fn = CumulativeLoss(weight=args.w_cumulative).to(device)
    cumulative_zero_loss_fn = CumulativeZeroLoss(threshold=0.1, weight=args.w_cumulative_zero).to(device)
    cumulative_sparsity_loss_fn = CumulativeSparsityLoss(weight=args.w_cumulative_sparsity).to(device)
    cumulative_activity_loss_fn = CumulativeActivityLoss(weight=args.w_cumulative_activity).to(device)
    cumulative_param_activity_loss_fn = CumulativeParamActivityLoss(weight=args.w_cumulative_param_activity).to(device)
    
    logger.debug(f"Pred cumulative min: {pred_cumulative.min().item():.4f}, max: {pred_cumulative.max().item():.4f}, has_nan: {torch.isnan(pred_cumulative).any().item()}")
    
    loss_cumulative = cumulative_loss_fn(pred_cumulative, cumulative_transforms)
    loss_cumulative_zero = cumulative_zero_loss_fn(pred_cumulative, cumulative_transforms, cumulative_param_activity)
    loss_cumulative_sparsity = cumulative_sparsity_loss_fn(pred_cumulative, cumulative_param_activity)
    loss_cumulative_activity, activity_f1_scores = cumulative_activity_loss_fn(cumulative_activity_logits, cumulative_activity)
    loss_cumulative_param_activity, param_activity_f1_scores = cumulative_param_activity_loss_fn(
        cumulative_param_activity_logits, cumulative_param_activity, cumulative_activity
    )
    
    logger.debug(f"Cumulative param activity logits min: {cumulative_param_activity_logits.min().item():.4f}, max: {cumulative_param_activity_logits.max().item():.4f}, has_nan: {torch.isnan(cumulative_param_activity_logits).any().item()}")
    logger.debug(f"Cumulative param activity labels min: {cumulative_param_activity.min().item():.4f}, max: {cumulative_param_activity.max().item():.4f}, has_nan: {torch.isnan(cumulative_param_activity).any().item()}")
    
    losses = {
        'loss_cumulative': loss_cumulative,
        'loss_cumulative_zero': loss_cumulative_zero,
        'loss_cumulative_sparsity': loss_cumulative_sparsity,
        'loss_cumulative_activity': loss_cumulative_activity,
        'loss_cumulative_param_activity': loss_cumulative_param_activity,
        'activity_f1_scores': activity_f1_scores,
        'param_activity_f1_scores': param_activity_f1_scores
    }
    for name, loss in losses.items():
        if isinstance(loss, torch.Tensor) and (torch.isnan(loss).any() or torch.isinf(loss).any()):
            logger.error(f"{name} is NaN or Inf: {loss.item()}")
    
    total_loss = (
        args.w_cumulative * loss_cumulative +
        args.w_cumulative_zero * loss_cumulative_zero +
        args.w_cumulative_sparsity * loss_cumulative_sparsity +
        args.w_cumulative_activity * loss_cumulative_activity +
        args.w_cumulative_param_activity * loss_cumulative_param_activity
    )
    
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        logger.error(f"Total loss is NaN or Inf: {total_loss.item()}")
    
    cumulative_activity_preds = (torch.sigmoid(cumulative_activity_logits) > 0.5).float()
    cumulative_activity_accuracy = (cumulative_activity_preds == cumulative_activity).float().mean()
    cumulative_param_activity_preds = (torch.sigmoid(cumulative_param_activity_logits) > 0.5).float()
    cumulative_param_activity_accuracy = (cumulative_param_activity_preds == cumulative_param_activity).float().mean()
    logger.debug(f"Cumulative activity prediction accuracy: {cumulative_activity_accuracy:.4f}")
    logger.debug(f"Cumulative param activity prediction accuracy: {cumulative_param_activity_accuracy:.4f}")
    
    zero_cumulative_pred = (torch.abs(pred_cumulative) < 0.01).float().mean()
    zero_cumulative_target = (cumulative_transforms == 0).float().mean()
    logger.debug(f"Zero Cumulative Pred: {zero_cumulative_pred:.4f}, Target: {zero_cumulative_target:.4f}")
    
    return total_loss, losses

def train(args):
    logger = setup_logging(args.log_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    torch.autograd.set_detect_anomaly(True)

    train_dataset = CumulativeJawTeethDataset(
        data_dir=args.data_dir,
        num_teeth=args.num_teeth,
        num_points=args.num_points,
        channels=args.channels,
        split='train',
        train_ratio=args.train_ratio,
        cache_dir=args.cache_dir,
        log_file=args.log_file
    )
    val_dataset = CumulativeJawTeethDataset(
        data_dir=args.data_dir,
        num_teeth=args.num_teeth,
        num_points=args.num_points,
        channels=args.channels,
        split='val',
        train_ratio=args.train_ratio,
        cache_dir=args.cache_dir,
        log_file=args.log_file
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    logger.info(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

    model = OrthoDGCNNModel(
        max_stages=args.max_stages,
        num_teeth=args.num_teeth,
        num_points=args.num_points,
        channels=args.channels,
        embed_dim=args.embed_dim,
        teacher_forcing=args.teacher_forcing_prob > 0,
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        k=args.k,
        decoder_type='per_tooth',  # New argument to select decoder type
        per_tooth_layers=args.per_tooth_layers,
        per_tooth_heads=args.per_tooth_heads,
        per_tooth_mlp_ratio=args.per_tooth_mlp_ratio
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        train_losses = {
            'total': 0.0,
            'cumulative': 0.0,
            'cumulative_zero': 0.0,
            'cumulative_sparsity': 0.0,
            'cumulative_activity': 0.0,
            'cumulative_param_activity': 0.0
        }
        for batch_idx, (jaw_id, feats, cumulative_transforms, cumulative_activity, cumulative_param_activity) in enumerate(train_loader):
            feats, cumulative_transforms, cumulative_activity, cumulative_param_activity = [
                x.to(device) for x in [feats, cumulative_transforms, cumulative_activity, cumulative_param_activity]
            ]
            
            logger.debug(f"Batch {batch_idx} shapes: feats={feats.shape}, "
                        f"cumulative_transforms={cumulative_transforms.shape}, "
                        f"cumulative_activity={cumulative_activity.shape}, "
                        f"cumulative_param_activity={cumulative_param_activity.shape}")
            
            optimizer.zero_grad()

            pred_cumulative, cumulative_activity_logits, cumulative_param_activity_logits = model(feats)

            total_loss, losses = compute_loss(
                pred_cumulative, cumulative_activity_logits, cumulative_param_activity_logits,
                cumulative_transforms, cumulative_activity, cumulative_param_activity,
                device, logger, args
            )

            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            logger.debug(f"Gradient norm: {grad_norm:.4f}")
            optimizer.step()
            scheduler.step()

            train_losses['total'] += total_loss.item()
            train_losses['cumulative'] += losses['loss_cumulative'].item()
            train_losses['cumulative_zero'] += losses['loss_cumulative_zero'].item()
            train_losses['cumulative_sparsity'] += losses['loss_cumulative_sparsity'].item()
            train_losses['cumulative_activity'] += losses['loss_cumulative_activity'].item()
            train_losses['cumulative_param_activity'] += losses['loss_cumulative_param_activity'].item()

            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}/{len(train_loader)}, "
                            f"Total Loss: {total_loss.item():.4f}, Cumulative: {losses['loss_cumulative'].item():.4f}, "
                            f"Cumulative_Zero: {losses['loss_cumulative_zero'].item():.4f}, "
                            f"Cumulative_Sparsity: {losses['loss_cumulative_sparsity'].item():.4f}, "
                            f"Cumulative_Activity: {losses['loss_cumulative_activity'].item():.4f}, "
                            f"Cumulative_Param_Activity: {losses['loss_cumulative_param_activity'].item():.4f}")

        for key in train_losses:
            train_losses[key] /= len(train_loader)

        model.eval()
        val_losses = {
            'total': 0.0,
            'cumulative': 0.0,
            'cumulative_zero': 0.0,
            'cumulative_sparsity': 0.0,
            'cumulative_activity': 0.0,
            'cumulative_param_activity': 0.0
        }
        val_activity_f1 = []
        val_param_activity_f1 = []
        with torch.no_grad():
            for jaw_id, feats, cumulative_transforms, cumulative_activity, cumulative_param_activity in val_loader:
                feats, cumulative_transforms, cumulative_activity, cumulative_param_activity = [
                    x.to(device) for x in [feats, cumulative_transforms, cumulative_activity, cumulative_param_activity]
                ]

                pred_cumulative, cumulative_activity_logits, cumulative_param_activity_logits = model(feats)

                total_loss, losses = compute_loss(
                    pred_cumulative, cumulative_activity_logits, cumulative_param_activity_logits,
                    cumulative_transforms, cumulative_activity, cumulative_param_activity,
                    device, logger, args
                )

                val_losses['total'] += total_loss.item()
                val_losses['cumulative'] += losses['loss_cumulative'].item()
                val_losses['cumulative_zero'] += losses['loss_cumulative_zero'].item()
                val_losses['cumulative_sparsity'] += losses['loss_cumulative_sparsity'].item()
                val_losses['cumulative_activity'] += losses['loss_cumulative_activity'].item()
                val_losses['cumulative_param_activity'] += losses['loss_cumulative_param_activity'].item()
                val_activity_f1.append(losses['activity_f1_scores'].cpu())
                val_param_activity_f1.append(losses['param_activity_f1_scores'].cpu())

        for key in val_losses:
            val_losses[key] /= len(val_loader)
        
        # Compute and log mean F1-scores for validation
        val_activity_f1 = torch.stack(val_activity_f1).mean(dim=0)  # [num_teeth]
        val_param_activity_f1 = torch.stack(val_param_activity_f1).mean(dim=0)  # [num_teeth, num_params]
        logger.info(f"Validation Activity F1 Scores (mean: {val_activity_f1.mean().item():.4f}): {val_activity_f1.tolist()}")
        logger.info(f"Validation Param Activity F1 Scores (mean: {val_param_activity_f1.mean().item():.4f})")

        logger.info(f"Epoch {epoch+1}/{args.epochs}, "
                    f"Train Loss: {train_losses['total']:.4f} (Cumulative: {train_losses['cumulative']:.4f}, "
                    f"Cumulative_Zero: {train_losses['cumulative_zero']:.4f}, "
                    f"Cumulative_Sparsity: {train_losses['cumulative_sparsity']:.4f}, "
                    f"Cumulative_Activity: {train_losses['cumulative_activity']:.4f}, "
                    f"Cumulative_Param_Activity: {train_losses['cumulative_param_activity']:.4f}), "
                    f"Val Loss: {val_losses['total']:.4f} (Cumulative: {val_losses['cumulative']:.4f}, "
                    f"Cumulative_Zero: {val_losses['cumulative_zero']:.4f}, "
                    f"Cumulative_Sparsity: {val_losses['cumulative_sparsity']:.4f}, "
                    f"Cumulative_Activity: {val_losses['cumulative_activity']:.4f}, "
                    f"Cumulative_Param_Activity: {val_losses['cumulative_param_activity']:.4f})")

        if val_losses['total'] < best_val_loss and epoch != 1:
            best_val_loss = val_losses['total']
            checkpoint_dgcnn = {
                'epoch': epoch + 1,
                'model_state_dict': model.dgcnn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss
            }
            checkpoint_cumulative = {
                'epoch': epoch + 1,
                'model_state_dict': model.cumulative_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss
            }
            torch.save(checkpoint_dgcnn, os.path.join(args.output_dir, 'best_dgcnn.pth'))
            torch.save(checkpoint_cumulative, os.path.join(args.output_dir, 'best_cumulative.pth'))
            logger.info(f"Saved best DGCNN model at epoch {epoch+1} with val_loss {best_val_loss:.4f}")
            logger.info(f"Saved best cumulative model at epoch {epoch+1} with val_loss {best_val_loss:.4f}")
        
        if epoch == (args.epochs - 1):
            last_val_losses = val_losses['total']
            checkpoint_dgcnn = {
                'epoch': epoch + 1,
                'model_state_dict': model.dgcnn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': last_val_losses
            }
            checkpoint_cumulative = {
                'epoch': epoch + 1,
                'model_state_dict': model.cumulative_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': last_val_losses
            }
            torch.save(checkpoint_dgcnn, os.path.join(args.output_dir, 'last_dgcnn.pth'))
            torch.save(checkpoint_cumulative, os.path.join(args.output_dir, 'last_cumulative.pth'))
            logger.info(f"Saved last DGCNN model at epoch {epoch+1} with val_loss {last_val_losses:.4f}")
            logger.info(f"Saved last cumulative model at epoch {epoch+1} with val_loss {last_val_losses:.4f}")

    logger.info("Training completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Orthodontic Cumulative Transformation Model")
    parser.add_argument('--data_dir', type=str, default='./Data', help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./output', help='Path to save checkpoints')
    parser.add_argument('--log_file', type=str, default='training_log.txt', help='Path to log file')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='Path to cache directory')
    parser.add_argument('--num_points', type=int, default=256, help='Number of points per tooth')
    parser.add_argument('--channels', type=int, default=4, help='Number of feature channels')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train/validation split ratio')
    parser.add_argument('--embed_dim', type=int, default=384, help='Embedding dimension')
    parser.add_argument('--per_tooth_layers', type=int, default=4, help='Number of layers in PerToothTransformerDecoder')
    parser.add_argument('--per_tooth_heads', type=int, default=8, help='Number of attention heads in PerToothTransformerDecoder')
    parser.add_argument('--per_tooth_mlp_ratio', type=float, default=4.0, help='MLP ratio in PerToothTransformerDecoder')
    parser.add_argument('--k', type=int, default=20, help='Number of k in DGCNN')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--num_teeth', type=int, default=14, help='Number of teeth')
    parser.add_argument('--w_cumulative', type=float, default=1.0, help='Weight for cumulative loss')
    parser.add_argument('--w_cumulative_zero', type=float, default=1.0, help='Weight for cumulative zero loss')
    parser.add_argument('--w_cumulative_sparsity', type=float, default=1.0, help='Weight for cumulative sparsity loss')
    parser.add_argument('--w_cumulative_activity', type=float, default=1.0, help='Weight for cumulative activity loss')
    parser.add_argument('--w_cumulative_param_activity', type=float, default=1.0, help='Weight for cumulative param activity loss')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    train(args)