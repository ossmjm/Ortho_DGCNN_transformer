import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import argparse
from torch.amp import autocast, GradScaler
from dataset import JawTeethDataset
from models.OrthoDGCNN_decoder import OrthoDGCNNModel
from losses_decoder import StagewiseMSELoss, ToothActivityLoss, ParamActivityLoss, PaddedLoss, ConsistencyLoss
from optimizers import Optimizers, LRSchedulers

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

def compute_loss(transforms_sequence, activity_logits, param_activity_logits, targets, activity_labels, param_activity_labels, cumulative_transforms, true_num_stages, max_stages, device, logger, args):
    mse_loss_fn = StagewiseMSELoss(weight=args.w_trans).to(device)
    activity_loss_fn = ToothActivityLoss(weight=args.w_activity).to(device)
    param_activity_loss_fn = ParamActivityLoss(weight=args.w_param_activity).to(device)
    padded_loss_fn = PaddedLoss(weight=args.w_padded).to(device)
    consistency_loss_fn = ConsistencyLoss(weight=args.w_consistency).to(device)
    
    transforms_sequence = torch.clamp(transforms_sequence, -5, 5)
    
    logger.debug(f"Transforms sequence min: {transforms_sequence.min().item():.4f}, max: {transforms_sequence.max().item():.4f}, has_nan: {torch.isnan(transforms_sequence).any().item()}")
    
    loss_mse = mse_loss_fn(transforms_sequence, targets, activity_labels.unsqueeze(-1))
    loss_activity, f1_activity = activity_loss_fn(activity_logits, activity_labels, true_num_stages)
    loss_param_activity, f1_param_activity = param_activity_loss_fn(param_activity_logits, param_activity_labels, activity_logits, true_num_stages)
    padded_loss = padded_loss_fn(transforms_sequence, true_num_stages, max_stages)
    consistency_loss = consistency_loss_fn(transforms_sequence, cumulative_transforms, true_num_stages, max_stages, device)
    
    losses = {
        'loss_mse': loss_mse,
        'loss_activity': loss_activity,
        'loss_param_activity': loss_param_activity,
        'padded_loss': padded_loss,
        'consistency_loss': consistency_loss
    }
    for name, loss in losses.items():
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"{name} is NaN or Inf: {loss.item()}")
    
    total_loss = (
        args.w_trans * loss_mse +
        args.w_activity * loss_activity +
        args.w_param_activity * loss_param_activity +
        args.w_padded * padded_loss +
        args.w_consistency * consistency_loss
    )
    
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        logger.error(f"Total loss is NaN or Inf: {total_loss.item()}")
    
    mean_f1_activity = f1_activity.mean().item()
    mean_f1_param_activity = f1_param_activity.mean().item()
    
    return total_loss, losses, mean_f1_activity, mean_f1_param_activity

def train(args):
    logger = setup_logging(args.log_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    torch.autograd.set_detect_anomaly(True)
    scaler = GradScaler()

    train_dataset = JawTeethDataset(
        data_dir=args.data_dir,
        max_stages=args.max_stages,
        num_teeth=args.num_teeth,
        num_points=args.num_points,
        channels=args.channels,
        split='train',
        train_ratio=args.train_ratio,
        cache_dir=args.cache_dir,
        log_file=args.log_file
    )
    val_dataset = JawTeethDataset(
        data_dir=args.data_dir,
        max_stages=args.max_stages,
        num_teeth=args.num_teeth,
        num_points=args.num_points,
        channels=args.channels,
        split='val',
        train_ratio=args.train_ratio,
        cache_dir=args.cache_dir,
        log_file=args.log_file
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    logger.info(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

    model = OrthoDGCNNModel(
        max_stages=args.max_stages,
        num_teeth=args.num_teeth,
        num_points=args.num_points,
        channels=args.channels,
        embed_dim=args.embed_dim,
        teacher_forcing_prob=args.teacher_forcing_prob,  # Pass float probability
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        k=args.k,
        decoder_type=args.decoder_type
    ).to(device)

    # Load model weights and initialize optimizers
    optimizer_dgcnn = Optimizers(
        optimizer_name=args.optimizer,
        parameters=model.dgcnn.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=args.optimizer_betas
    ).get_optimizer()
    optimizer_decoder = Optimizers(
        optimizer_name=args.optimizer,
        parameters=model.decoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=args.optimizer_betas
    ).get_optimizer()

    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        try:
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['ortho_dgcnn_state_dict'])
            logger.info(f"Loaded full model weights from {args.checkpoint_path}")
            # Attempt to load optimizer states
            if 'optimizer_dgcnn_state_dict' in checkpoint and checkpoint['optimizer_dgcnn_state_dict'] is not None:
                try:
                    optimizer_dgcnn.load_state_dict(checkpoint['optimizer_dgcnn_state_dict'])
                    logger.info("Loaded optimizer_dgcnn state from checkpoint")
                except Exception as e:
                    logger.warning(f"Failed to load optimizer_dgcnn state: {e}. Resetting optimizer state.")
                    optimizer_dgcnn.state = {}  # Reset state to ensure clean initialization
            if 'optimizer_decoder_state_dict' in checkpoint and checkpoint['optimizer_decoder_state_dict'] is not None:
                try:
                    optimizer_decoder.load_state_dict(checkpoint['optimizer_decoder_state_dict'])
                    logger.info("Loaded optimizer_decoder state from checkpoint")
                except Exception as e:
                    logger.warning(f"Failed to load optimizer_decoder state: {e}. Resetting optimizer state.")
                    optimizer_decoder.state = {}  # Reset state to ensure clean initialization
        except Exception as e:
            logger.error(f"Failed to load model weights from {args.checkpoint_path}: {e}")
            raise
    else:
        checkpoint = torch.load(args.pretrained_dgcnn_path, map_location=device)
        model.dgcnn.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded pretrained DGCNN weights from {args.pretrained_dgcnn_path}")

    # Initialize schedulers only if use_scheduler is True
    scheduler_dgcnn = LRSchedulers(
        scheduler_name=args.scheduler,
        optimizer=optimizer_dgcnn,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        warmup_start_factor=args.warmup_start_factor,
        eta_min=args.scheduler_eta_min,
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        use_scheduler=args.use_scheduler
    ).get_scheduler()
    scheduler_decoder = LRSchedulers(
        scheduler_name=args.scheduler,
        optimizer=optimizer_decoder,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        warmup_start_factor=args.warmup_start_factor,
        eta_min=args.scheduler_eta_min,
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        use_scheduler=args.use_scheduler
    ).get_scheduler()

    # Load scheduler states if available and scheduler is used
    if args.checkpoint_path and os.path.exists(args.checkpoint_path) and args.use_scheduler:
        try:
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            if 'scheduler_dgcnn_state_dict' in checkpoint and checkpoint['scheduler_dgcnn_state_dict'] is not None and scheduler_dgcnn is not None:
                try:
                    scheduler_dgcnn.load_state_dict(checkpoint['scheduler_dgcnn_state_dict'])
                    logger.info("Loaded scheduler_dgcnn state from checkpoint")
                except Exception as e:
                    logger.warning(f"Failed to load scheduler_dgcnn state: {e}. Using new scheduler.")
            if 'scheduler_decoder_state_dict' in checkpoint and checkpoint['scheduler_decoder_state_dict'] is not None and scheduler_decoder is not None:
                try:
                    scheduler_decoder.load_state_dict(checkpoint['scheduler_decoder_state_dict'])
                    logger.info("Loaded scheduler_decoder state from checkpoint")
                except Exception as e:
                    logger.warning(f"Failed to load scheduler_decoder state: {e}. Using new scheduler.")
        except Exception as e:
            logger.warning(f"Failed to load scheduler states from {args.checkpoint_path}: {e}")

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_losses = {
            'total': 0.0,
            'loss_mse': 0.0,
            'loss_activity': 0.0,
            'loss_param_activity': 0.0,
            'padded_loss': 0.0,
            'consistency_loss': 0.0
        }
        train_f1_activity = []
        train_f1_param_activity = []

        for batch_idx, (jaw_id, feats, transforms, cumulative_transforms, activity, param_activity, _, _, _, num_stages) in enumerate(train_loader):
            feats, transforms, cumulative_transforms, activity, param_activity, num_stages = [
                x.to(device) for x in [feats, transforms, cumulative_transforms, activity, param_activity, num_stages]
            ]
            
            optimizer_dgcnn.zero_grad()
            optimizer_decoder.zero_grad()

            with autocast(device_type='cuda'):
                pred_transforms, activity_logits, param_activity_logits = model(
                    coordinates=feats,
                    targets=transforms,
                    cumulative_targets=cumulative_transforms,
                    activity_targets=activity,
                    param_activity_targets=param_activity,
                    num_stages=num_stages,
                    epoch=epoch,
                    total_epochs=args.epochs
                )

                total_loss, losses, mean_f1_activity, mean_f1_param_activity = compute_loss(
                    pred_transforms, activity_logits, param_activity_logits,
                    transforms, activity, param_activity, cumulative_transforms,
                    num_stages, args.max_stages, device, logger, args
                )

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer_dgcnn)
            scaler.unscale_(optimizer_decoder)
            torch.nn.utils.clip_grad_norm_(model.dgcnn.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=0.5)
            scaler.step(optimizer_dgcnn)
            scaler.step(optimizer_decoder)
            scaler.update()

            train_losses['total'] += total_loss.item()
            for key in losses:
                train_losses[key] += losses[key].item()
            train_f1_activity.append(mean_f1_activity)
            train_f1_param_activity.append(mean_f1_param_activity)

            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}/{len(train_loader)}, "
                            f"Total Loss: {total_loss.item():.4f}, MSE: {losses['loss_mse'].item():.4f}, "
                            f"Activity: {losses['loss_activity'].item():.4f}, Param Activity: {losses['loss_param_activity'].item():.4f}, "
                            f"Padded: {losses['padded_loss'].item():.4f}, "
                            f"Consistency: {losses['consistency_loss'].item():.4f}")

        # Step schedulers only if they exist and scheduler type is not reduceonplateau
        if args.use_scheduler and args.scheduler.lower() != 'reduceonplateau':
            if scheduler_dgcnn is not None:
                scheduler_dgcnn.step()
            if scheduler_decoder is not None:
                scheduler_decoder.step()

        for key in train_losses:
            train_losses[key] /= len(train_loader)
        mean_train_f1_activity = np.mean(train_f1_activity)
        mean_train_f1_param_activity = np.mean(train_f1_param_activity)

        model.eval()
        val_losses = {
            'total': 0.0,
            'loss_mse': 0.0,
            'loss_activity': 0.0,
            'loss_param_activity': 0.0,
            'padded_loss': 0.0,
            'consistency_loss': 0.0
        }
        val_f1_activity = []
        val_f1_param_activity = []

        with torch.no_grad():
            for jaw_id, feats, transforms, cumulative_transforms, activity, param_activity, _, _, _, num_stages in val_loader:
                feats, transforms, cumulative_transforms, activity, param_activity, num_stages = [
                    x.to(device) for x in [feats, transforms, cumulative_transforms, activity, param_activity, num_stages]
                ]

                with autocast(device_type='cuda'):
                    pred_transforms, activity_logits, param_activity_logits = model(
                        coordinates=feats,
                        targets=transforms,
                        cumulative_targets=cumulative_transforms,
                        activity_targets=activity,
                        param_activity_targets=param_activity,
                        num_stages=num_stages,
                        epoch=epoch,
                        total_epochs=args.epochs
                    )

                    total_loss, losses, mean_f1_activity, mean_f1_param_activity = compute_loss(
                        pred_transforms, activity_logits, param_activity_logits,
                        transforms, activity, param_activity, cumulative_transforms,
                        num_stages, args.max_stages, device, logger, args
                    )

                val_losses['total'] += total_loss.item()
                for key in losses:
                    val_losses[key] += losses[key].item()
                val_f1_activity.append(mean_f1_activity)
                val_f1_param_activity.append(mean_f1_param_activity)

        for key in val_losses:
            val_losses[key] /= len(val_loader)
        mean_val_f1_activity = np.mean(val_f1_activity)
        mean_val_f1_param_activity = np.mean(val_f1_param_activity)

        # Step schedulers for reduceonplateau only if scheduler is used
        if args.use_scheduler and args.scheduler.lower() == 'reduceonplateau':
            if scheduler_dgcnn is not None:
                scheduler_dgcnn.step(val_losses['total'])
            if scheduler_decoder is not None:
                scheduler_decoder.step(val_losses['total'])

        logger.info(f"Epoch {epoch+1}/{args.epochs}, "
                    f"Train Loss: {train_losses['total']:.4f}, MSE: {train_losses['loss_mse']:.4f}, "
                    f"Activity: {train_losses['loss_activity']:.4f}, Param Activity: {train_losses['loss_param_activity']:.4f}, "
                    f"Padded: {train_losses['padded_loss']:.4f}, Consistency: {train_losses['consistency_loss']:.4f}, "
                    f"Train Mean F1 Activity: {mean_train_f1_activity:.4f}, Train Mean F1 Param Activity: {mean_train_f1_param_activity:.4f}")
        
        logger.info(f"Epoch {epoch+1}/{args.epochs}, "
                    f"Val Loss: {val_losses['total']:.4f}, MSE: {val_losses['loss_mse']:.4f}, "
                    f"Activity: {val_losses['loss_activity']:.4f}, Param Activity: {val_losses['loss_param_activity']:.4f}, "
                    f"Padded: {val_losses['padded_loss']:.4f}, Consistency: {val_losses['consistency_loss']:.4f}, "
                    f"Val Mean F1 Activity: {mean_val_f1_activity:.4f}, Val Mean F1 Param Activity: {mean_val_f1_param_activity:.4f}")

        if epoch % 10 == 0 and epoch != 0:
            # Save DGCNN model separately
            dgcnn_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.dgcnn.state_dict(),
                'optimizer_dgcnn_state_dict': optimizer_dgcnn.state_dict(),
                'scheduler_dgcnn_state_dict': scheduler_dgcnn.state_dict() if args.use_scheduler and scheduler_dgcnn is not None else None,
                'val_loss': val_losses['total']
            }
            torch.save(dgcnn_checkpoint, os.path.join(args.output_dir, f'dgcnn_epoch_{epoch + 1}.pth'))
            logger.info(f"Saved DGCNN model at epoch {epoch + 1} with val_loss {val_losses['total']:.4f}")

            # Save whole model
            checkpoint = {
                'epoch': epoch + 1,
                'dgcnn_state_dict': model.dgcnn.state_dict(),
                'decoder_state_dict': model.decoder.state_dict(),
                'ortho_dgcnn_state_dict': model.state_dict(),
                'optimizer_dgcnn_state_dict': optimizer_dgcnn.state_dict(),
                'optimizer_decoder_state_dict': optimizer_decoder.state_dict(),
                'scheduler_dgcnn_state_dict': scheduler_dgcnn.state_dict() if args.use_scheduler and scheduler_dgcnn is not None else None,
                'scheduler_decoder_state_dict': scheduler_decoder.state_dict() if args.use_scheduler and scheduler_decoder is not None else None,
                'val_loss': val_losses['total']
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f'model_epoch_{epoch + 1}.pth'))
            logger.info(f"Saved full model at epoch {epoch + 1} with val_loss {val_losses['total']:.4f}")

        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"No improvement in val_loss, patience counter: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            # Save DGCNN model separately
            dgcnn_checkpoint = {
                'epoch': best_epoch,
                'model_state_dict': model.dgcnn.state_dict(),
                'optimizer_dgcnn_state_dict': optimizer_dgcnn.state_dict(),
                'scheduler_dgcnn_state_dict': scheduler_dgcnn.state_dict() if args.use_scheduler and scheduler_dgcnn is not None else None,
                'val_loss': best_val_loss
            }
            torch.save(dgcnn_checkpoint, os.path.join(args.output_dir, f'best_dgcnn.pth'))
            logger.info(f"Saved best DGCNN model at epoch {best_epoch} with val_loss {best_val_loss:.4f}")

            # Save whole model
            checkpoint = {
                'epoch': best_epoch,
                'dgcnn_state_dict': model.dgcnn.state_dict(),
                'decoder_state_dict': model.decoder.state_dict(),
                'ortho_dgcnn_state_dict': model.state_dict(),
                'optimizer_dgcnn_state_dict': optimizer_dgcnn.state_dict(),
                'optimizer_decoder_state_dict': optimizer_decoder.state_dict(),
                'scheduler_dgcnn_state_dict': scheduler_dgcnn.state_dict() if args.use_scheduler and scheduler_dgcnn is not None else None,
                'scheduler_decoder_state_dict': scheduler_decoder.state_dict() if args.use_scheduler and scheduler_decoder is not None else None,
                'val_loss': best_val_loss
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f'best_model.pth'))
            logger.info(f"Saved best full model at epoch {best_epoch} with val_loss {best_val_loss:.4f}")
            logger.info(f"Early stopping triggered at epoch {epoch+1} after {args.patience} epochs without improvement")
            break

        if epoch == args.epochs - 1:
            last_val_loss = val_losses['total']
            # Save DGCNN model separately
            dgcnn_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.dgcnn.state_dict(),
                'optimizer_dgcnn_state_dict': optimizer_dgcnn.state_dict(),
                'scheduler_dgcnn_state_dict': scheduler_dgcnn.state_dict() if args.use_scheduler and scheduler_dgcnn is not None else None,
                'val_loss': last_val_loss
            }
            torch.save(dgcnn_checkpoint, os.path.join(args.output_dir, f'last_dgcnn.pth'))
            logger.info(f"Saved last DGCNN model at epoch {epoch + 1} with val_loss {last_val_loss:.4f}")

            # Save whole model
            checkpoint = {
                'epoch': epoch + 1,
                'dgcnn_state_dict': model.dgcnn.state_dict(),
                'decoder_state_dict': model.decoder.state_dict(),
                'ortho_dgcnn_state_dict': model.state_dict(),
                'optimizer_dgcnn_state_dict': optimizer_dgcnn.state_dict(),
                'optimizer_decoder_state_dict': optimizer_decoder.state_dict(),
                'scheduler_dgcnn_state_dict': scheduler_dgcnn.state_dict() if args.use_scheduler and scheduler_dgcnn is not None else None,
                'scheduler_decoder_state_dict': scheduler_decoder.state_dict() if args.use_scheduler and scheduler_dgcnn is not None else None,
                'val_loss': last_val_loss
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'last_model.pth'))
            logger.info(f"Saved last full model at epoch {epoch + 1} with val_loss {last_val_loss:.4f}")

    logger.info("Training completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Orthodontic Treatment Prediction Model")
    parser.add_argument('--data_dir', type=str, default='./Data', help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./output_decoder_l', help='Path to save checkpoints')
    parser.add_argument('--log_file', type=str, default='training_log_decoder_l.txt', help='Path to log file')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='Path to cache directory')
    parser.add_argument('--pretrained_dgcnn_path', type=str, default='./output/best_dgcnn.pth', help='Path to pretrained DGCNN weights')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint of the whole model (optional)')
    parser.add_argument('--num_points', type=int, default=256, help='Number of points per tooth')
    parser.add_argument('--channels', type=int, default=4, help='Number of feature channels')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train/validation split ratio')
    parser.add_argument('--embed_dim', type=int, default=96, help='Embedding dimension')
    parser.add_argument('--k', type=int, default=10, help='Number of k in DGCNN')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help='MLP ratio in Transformer')
    parser.add_argument('--decoder_layers', type=int, default=1, help='Number of decoder layers in Transformer')
    parser.add_argument('--decoder_type', type=str, default='decoder', help='Type of used decoder model')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--teacher_forcing_prob', type=float, default=0.8, help='Teacher forcing probability')
    parser.add_argument('--max_stages', type=int, default=25, help='Maximum number of stages')
    parser.add_argument('--num_teeth', type=int, default=14, help='Number of teeth')
    parser.add_argument('--w_trans', type=float, default=1.0, help='Weight for MSE loss')
    parser.add_argument('--w_activity', type=float, default=1.0, help='Weight for activity loss')
    parser.add_argument('--w_param_activity', type=float, default=1.0, help='Weight for param activity loss')
    parser.add_argument('--w_padded', type=float, default=0.2, help='Weight for padded loss')
    parser.add_argument('--w_consistency', type=float, default=0.2, help='Weight for consistency loss')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'radam', 'lion', 'sparseadam', 'adan', 'caadam'],
                        help='Optimizer type')
    parser.add_argument('--optimizer_betas', type=float, nargs=2, default=[0.9, 0.999], help='Betas for optimizers')
    parser.add_argument('--scheduler', type=str, default='cosineannealing', choices=['cosineannealing', 'reduceonplateau', 'linear'],
                        help='Learning rate scheduler type')
    parser.add_argument('--scheduler_eta_min', type=float, default=0.0, help='Minimum learning rate for CosineAnnealing')
    parser.add_argument('--scheduler_factor', type=float, default=0.5, help='Factor for ReduceLROnPlateau')
    parser.add_argument('--scheduler_patience', type=int, default=5, help='Patience for ReduceLROnPlateau')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Number of warmup epochs')
    parser.add_argument('--warmup_start_factor', type=float, default=0.1, help='Starting factor for warmup')
    parser.add_argument('--use_scheduler', type=bool, default=False, help='Whether to use a learning rate scheduler')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    train(args)