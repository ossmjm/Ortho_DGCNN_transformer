import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import argparse
from dataset import CumulativeJawTeethDataset
from models.OrthoDGCNN import OrthoDGCNNModel
from losses_cumulative import compute_loss
import numpy as np

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
        log_file=args.log_file,
        use_scaler=args.use_scaler,
        scaler_type=args.scaler_type
    )
    val_dataset = CumulativeJawTeethDataset(
        data_dir=args.data_dir,
        num_teeth=args.num_teeth,
        num_points=args.num_points,
        channels=args.channels,
        split='val',
        train_ratio=args.train_ratio,
        cache_dir=args.cache_dir,
        log_file=args.log_file,
        use_scaler=args.use_scaler,
        scaler_type=args.scaler_type
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    logger.info(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

    model = OrthoDGCNNModel(
        num_teeth=args.num_teeth,
        num_points=args.num_points,
        channels=args.channels,
        embed_dim=args.embed_dim,
        k=args.k,
        encoder_type=args.encoder_type
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_loss = float('inf')
    patience_counter = 0

    train_loss_history = {
        'total': [],
        'loss_cumulative': [],
        'loss_cumulative_activity': [],
        'loss_cumulative_param_activity': [],
        'loss_cumulative_direction': [],
        'activity_f1_scores': [],
        'param_activity_f1_scores': [],
        'direction_f1_scores': []
    }
    val_loss_history = {
        'total': [],
        'loss_cumulative': [],
        'loss_cumulative_activity': [],
        'loss_cumulative_param_activity': [],
        'loss_cumulative_direction': [],
        'activity_f1_scores': [],
        'param_activity_f1_scores': [],
        'direction_f1_scores': []
    }

    for epoch in range(args.epochs):
        model.train()
        train_losses = {
            'total': 0.0,
            'loss_cumulative': 0.0,
            'loss_cumulative_activity': 0.0,
            'loss_cumulative_param_activity': 0.0,
            'loss_cumulative_direction': 0.0,
            'activity_f1_scores': 0.0,
            'param_activity_f1_scores': 0.0,
            'direction_f1_scores': 0.0
        }
        for batch_idx, (jaw_id, feats, cumulative_transforms, cumulative_activity, cumulative_param_activity, directions) in enumerate(train_loader):
            feats, cumulative_transforms, cumulative_activity, cumulative_param_activity, directions = [
                x.to(device) for x in [feats, cumulative_transforms, cumulative_activity, cumulative_param_activity, directions]
            ]
            
            logger.debug(f"Batch {batch_idx} shapes: feats={feats.shape}, "
                        f"cumulative_transforms={cumulative_transforms.shape}, "
                        f"cumulative_activity={cumulative_activity.shape}, "
                        f"cumulative_param_activity={cumulative_param_activity.shape}, "
                        f"directions={directions.shape}")
            
            optimizer.zero_grad()

            pred_cumulative, cumulative_activity_logits, cumulative_param_activity_logits, directions_logits = model(feats)

            total_loss, losses = compute_loss(
                pred_cumulative, cumulative_activity_logits, cumulative_param_activity_logits, directions_logits,
                cumulative_transforms, cumulative_activity, cumulative_param_activity, directions,
                device, logger, args
            )

            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            logger.debug(f"Gradient norm: {grad_norm:.4f}")
            optimizer.step()
            scheduler.step()

            train_losses['total'] += total_loss.item()
            for key in losses:
                if key.endswith('_f1_scores'):
                    train_losses[key] += losses[key].mean().item()  # Average F1 scores
                else:
                    train_losses[key] += losses[key].item()

            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}/{len(train_loader)}, "
                            f"Total Loss: {total_loss.item():.4f}, "
                            f"Cumulative: {losses['loss_cumulative'].item():.4f}, "
                            f"Cumulative_Activity: {losses['loss_cumulative_activity'].item():.4f}, "
                            f"Cumulative_Param_Activity: {losses['loss_cumulative_param_activity'].item():.4f}, "
                            f"Cumulative_Direction: {losses['loss_cumulative_direction'].item():.4f}, "
                            f"Activity_F1: {losses['activity_f1_scores'].mean().item():.4f}, "
                            f"Param_Activity_F1: {losses['param_activity_f1_scores'].mean().item():.4f}, "
                            f"Direction_F1: {losses['direction_f1_scores'].mean().item():.4f}")

        for key in train_losses:
            train_losses[key] /= len(train_loader)
        
        train_loss_history['total'].append(train_losses['total'])
        for key in losses:
            train_loss_history[key].append(train_losses[key])

        model.eval()
        val_losses = {
            'total': 0.0,
            'loss_cumulative': 0.0,
            'loss_cumulative_activity': 0.0,
            'loss_cumulative_param_activity': 0.0,
            'loss_cumulative_direction': 0.0,
            'activity_f1_scores': 0.0,
            'param_activity_f1_scores': 0.0,
            'direction_f1_scores': 0.0
        }
        with torch.no_grad():
            for jaw_id, feats, cumulative_transforms, cumulative_activity, cumulative_param_activity, directions in val_loader:
                feats, cumulative_transforms, cumulative_activity, cumulative_param_activity, directions = [
                    x.to(device) for x in [feats, cumulative_transforms, cumulative_activity, cumulative_param_activity, directions]
                ]

                pred_cumulative, cumulative_activity_logits, cumulative_param_activity_logits, directions_logits = model(feats)

                total_loss, losses = compute_loss(
                    pred_cumulative, cumulative_activity_logits, cumulative_param_activity_logits, directions_logits,
                    cumulative_transforms, cumulative_activity, cumulative_param_activity, directions,
                    device, logger, args
                )
            
                val_losses['total'] += total_loss.item()
                for key in losses:
                    if key.endswith('_f1_scores'):
                        val_losses[key] += losses[key].mean().item()  # Average F1 scores
                    else:
                        val_losses[key] += losses[key].item()

        for key in val_losses:
            val_losses[key] /= len(val_loader)
        
        val_loss_history['total'].append(val_losses['total'])
        for key in losses:
            val_loss_history[key].append(val_losses[key])

        logger.info(f"Epoch {epoch+1}/{args.epochs}, "
                    f"Train Loss: {train_losses['total']:.4f} (Cumulative: {train_losses['loss_cumulative']:.4f}, "
                    f"Cumulative_Activity: {train_losses['loss_cumulative_activity']:.4f}, "
                    f"Cumulative_Param_Activity: {train_losses['loss_cumulative_param_activity']:.4f}, "
                    f"Cumulative_Direction: {train_losses['loss_cumulative_direction']:.4f}, "
                    f"Activity_F1: {train_losses['activity_f1_scores']:.4f}, "
                    f"Param_Activity_F1: {train_losses['param_activity_f1_scores']:.4f}, "
                    f"Direction_F1: {train_losses['direction_f1_scores']:.4f})")
        
        logger.info(f"Epoch {epoch+1}/{args.epochs}, "
                    f"Val Loss: {val_losses['total']:.4f} (Cumulative: {val_losses['loss_cumulative']:.4f}, "
                    f"Cumulative_Activity: {val_losses['loss_cumulative_activity']:.4f}, "
                    f"Cumulative_Param_Activity: {val_losses['loss_cumulative_param_activity']:.4f}, "
                    f"Cumulative_Direction: {val_losses['loss_cumulative_direction']:.4f}, "
                    f"Activity_F1: {val_losses['activity_f1_scores']:.4f}, "
                    f"Param_Activity_F1: {val_losses['param_activity_f1_scores']:.4f}, "
                    f"Direction_F1: {val_losses['direction_f1_scores']:.4f})")

        if (epoch + 1) % 5 == 0 and epoch != 0:
            checkpoint = {
                'epoch': epoch + 1,
                'encoder_state_dict': model.encoder.state_dict(),
                'model_state_dict': model.cumulative_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_losses['total']
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f'model_epoch_{epoch + 1}.pth'))
            logger.info(f"Saved full model at epoch {epoch+1} with val_loss {val_losses['total']:.4f}")

            for key in train_loss_history:
                np.save(os.path.join(args.output_dir, f'train_{key}_history.npy'), np.array(train_loss_history[key]))
                logger.info(f"Saved train {key} history")
            for key in val_loss_history:
                np.save(os.path.join(args.output_dir, f'val_{key}_history.npy'), np.array(val_loss_history[key]))
                logger.info(f"Saved validation {key} history")

        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"No improvement in val_loss, patience counter: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            checkpoint = {
                'epoch': epoch + 1,
                'encoder_state_dict': model.encoder.state_dict(),
                'model_state_dict': model.cumulative_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f'best_model.pth'))
            for key in train_loss_history:
                np.save(os.path.join(args.output_dir, f'train_{key}_history.npy'), np.array(train_loss_history[key]))
                logger.info(f"Saved train {key} history")
            for key in val_loss_history:
                np.save(os.path.join(args.output_dir, f'val_{key}_history.npy'), np.array(val_loss_history[key]))
                logger.info(f"Saved validation {key} history")

            logger.info(f"Saved best full model at epoch {best_epoch} with val_loss {best_val_loss:.4f}")
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

        if epoch == args.epochs - 1:
            last_val_loss = val_losses['total']
            checkpoint = {
                'epoch': epoch + 1,
                'encoder_state_dict': model.encoder.state_dict(),
                'model_state_dict': model.cumulative_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': last_val_loss
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'last_model.pth'))
            logger.info(f"Saved last full model at epoch {epoch + 1} with val_loss {last_val_loss:.4f}")

    for key in train_loss_history:
        np.save(os.path.join(args.output_dir, f'train_{key}_history.npy'), np.array(train_loss_history[key]))
        logger.info(f"Saved train {key} history")
    
    for key in val_loss_history:
        np.save(os.path.join(args.output_dir, f'val_{key}_history.npy'), np.array(val_loss_history[key]))
        logger.info(f"Saved validation {key} history")
    logger.info("Training completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Orthodontic Cumulative Transformation Model")
    parser.add_argument('--data_dir', type=str, default='./Data', help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./output', help='Path to save checkpoints')
    parser.add_argument('--log_file', type=str, default='training_log.txt', help='Path to log file')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='Path to cache directory')
    parser.add_argument('--num_points', type=int, default=256, help='Number of points per tooth')
    parser.add_argument('--channels', type=int, default=3, help='Number of feature channels')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train/validation split ratio')
    parser.add_argument('--use_scaler', type=bool, default=False, help='Apply scaler to transformations')
    parser.add_argument('--scaler_type', type=str, default='robust', choices=['robust', 'standard'], help='Scaler type')
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--encoder_type', type=str, default='dgcnn', help='Encoder type (DGCNN or pointnet++)')
    parser.add_argument('--k', type=int, default=20, help='Number of k in DGCNN')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--num_teeth', type=int, default=14, help='Number of teeth')
    parser.add_argument('--w_cumulative', type=float, default=1.0, help='Weight for cumulative loss')
    parser.add_argument('--w_cumulative_activity', type=float, default=1.0, help='Weight for cumulative activity loss')
    parser.add_argument('--w_cumulative_param_activity', type=float, default=1.0, help='Weight for cumulative param activity loss')
    parser.add_argument('--w_cumulative_direction', type=float, default=1.0, help='Weight for cumulative direction loss')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    train(args)