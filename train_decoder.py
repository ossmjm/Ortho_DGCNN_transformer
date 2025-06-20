import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import argparse
from dataset import JawTeethDataset
from models.OrthoDGCNN_decoder import OrthoDGCNNModel
from losses_decoder import HybridTransformLoss, ToothActivityLoss, ParamActivityLoss, PaddedLoss, ConsistencyLoss, StageActivityLoss
from optimizers import Optimizers, LRSchedulers
from torchmetrics.functional.classification import binary_f1_score

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True

def setup_logging(log_file):
    logger = logging.getLogger('TrainLogger')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def compute_loss(transforms_sequence, activity_logits, param_activity_logits, stage_activity_logits, targets, activity_labels, param_activity_labels, cumulative_transforms, true_num_stages, max_stages, device, logger, args):
    mse_loss_fn = HybridTransformLoss(
        weight=args.w_trans,
        small_error_threshold=args.small_error_threshold,
        large_delta=args.large_delta,
        sparse_weight=args.sparse_weight,
        max_error=args.max_error,
        small_error_scale=args.small_error_scale
    ).to(device)
    activity_loss_fn = ToothActivityLoss(weight=args.w_activity, use_focal=args.use_focal_loss, alpha=args.focal_alpha, gamma=args.focal_gamma).to(device)
    param_activity_loss_fn = ParamActivityLoss(weight=args.w_param_activity, use_focal=args.use_focal_loss, alpha=args.focal_alpha, gamma=args.focal_gamma).to(device)
    padded_loss_fn = PaddedLoss(weight=args.w_padded).to(device)
    consistency_loss_fn = ConsistencyLoss(weight=args.w_consistency).to(device)
    stage_activity_loss_fn = StageActivityLoss(weight=args.w_stage_activity, seq_penalty=args.seq_penalty).to(device)
    if args.use_scaler:
        train_dataset = JawTeethDataset(
            data_dir=args.data_dir,
            max_stages=args.max_stages,
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
        scalers = train_dataset.get_scalers()
    else:
        scalers = None
    # Warn if any loss weight is zero
    for w_name, w_value in [
        ('w_trans', args.w_trans), ('w_activity', args.w_activity), ('w_param_activity', args.w_param_activity),
        ('w_padded', args.w_padded), ('w_consistency', args.w_consistency), ('w_stage_activity', args.w_stage_activity)
    ]:
        if w_value == 0.0:
            logger.warning(f"Loss weight {w_name} is set to 0.0; corresponding loss will not contribute to training.")

    transforms_sequence = torch.clamp(transforms_sequence, 0, 40)
    
    loss_mse = mse_loss_fn(transforms_sequence, targets, activity_labels.unsqueeze(-1))
    loss_activity, f1_activity = activity_loss_fn(activity_logits, activity_labels, true_num_stages)
    loss_param_activity, f1_param_activity = param_activity_loss_fn(param_activity_logits, param_activity_labels, activity_labels, true_num_stages)
    loss_stage_activity, f1_stage_activity = stage_activity_loss_fn(stage_activity_logits, true_num_stages)
    
    padded_loss = padded_loss_fn(transforms_sequence, true_num_stages, max_stages)
    consistency_loss = consistency_loss_fn(transforms_sequence, cumulative_transforms, true_num_stages, max_stages, device,use_scaler=args.use_scaler,scalers=scalers)
    
    losses = {
        'loss_mse': loss_mse,
        'loss_activity': loss_activity,
        'loss_param_activity': loss_param_activity,
        'padded_loss': padded_loss,
        'consistency_loss': consistency_loss,
        'loss_stage_activity': loss_stage_activity
    }
    for name, loss in losses.items():
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"{name} is NaN or Inf: {loss.item()}")
    
    total_loss = (
        args.w_trans * loss_mse +
        args.w_activity * loss_activity +
        args.w_param_activity * loss_param_activity +
        args.w_padded * padded_loss +
        args.w_consistency * consistency_loss +
        args.w_stage_activity * loss_stage_activity
    )
    
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        logger.error(f"Total loss is NaN or Inf: {total_loss.item()}")
    
    return total_loss, losses, f1_activity, f1_param_activity, f1_stage_activity

def train(args):
    logger = setup_logging(args.log_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}, training in FP32")

    torch.autograd.set_detect_anomaly(True)

    train_dataset = JawTeethDataset(
        data_dir=args.data_dir,
        max_stages=args.max_stages,
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
    val_dataset = JawTeethDataset(
        data_dir=args.data_dir,
        max_stages=args.max_stages,
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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    logger.info(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

    model = OrthoDGCNNModel(
        max_stages=args.max_stages,
        num_teeth=args.num_teeth,
        num_points=args.num_points,
        channels=args.channels,
        embed_dim=args.embed_dim,
        teacher_forcing_prob=args.teacher_forcing_prob,
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        k=args.k,
        decoder_type=args.decoder_type
    ).to(device)

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
            if 'optimizer_dgcnn_state_dict' in checkpoint and checkpoint['optimizer_dgcnn_state_dict'] is not None:
                try:
                    optimizer_dgcnn.load_state_dict(checkpoint['optimizer_dgcnn_state_dict'])
                    logger.info("Loaded optimizer_dgcnn state from checkpoint")
                except Exception as e:
                    logger.warning(f"Failed to load optimizer_dgcnn state: {e}. Resetting optimizer state.")
                    optimizer_dgcnn.state = {}
            if 'optimizer_decoder_state_dict' in checkpoint and checkpoint['optimizer_decoder_state_dict'] is not None:
                try:
                    optimizer_decoder.load_state_dict(checkpoint['optimizer_decoder_state_dict'])
                    logger.info("Loaded optimizer_decoder state from checkpoint")
                except Exception as e:
                    logger.warning(f"Failed to load optimizer_decoder state: {e}. Resetting optimizer state.")
                    optimizer_decoder.state = {}
        except Exception as e:
            logger.error(f"Failed to load model weights from {args.checkpoint_path}: {e}")
            raise
    elif args.pretrained_dgcnn_path and os.path.exists(args.pretrained_dgcnn_path):
        checkpoint = torch.load(args.pretrained_dgcnn_path, map_location=device)
        model.dgcnn.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded pretrained DGCNN weights from {args.pretrained_dgcnn_path}")

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

    train_loss_history = {
        'total': [],
        'loss_mse': [],
        'loss_activity': [],
        'loss_param_activity': [],
        'padded_loss': [],
        'consistency_loss': [],
        'loss_stage_activity': [],
        'mean_f1_activity': [],
        'mean_f1_param_activity': [],
        'mean_f1_stage_activity': []
    }
    val_loss_history = {
        'total': [],
        'loss_mse': [],
        'loss_activity': [],
        'loss_param_activity': [],
        'padded_loss': [],
        'consistency_loss': [],
        'loss_stage_activity': [],
        'mean_f1_activity': [],
        'mean_f1_param_activity': [],
        'mean_f1_stage_activity': []
    }
    for epoch in range(args.epochs):
        model.train()
        train_losses = {
            'total': 0.0,
            'loss_mse': 0.0,
            'loss_activity': 0.0,
            'loss_param_activity': 0.0,
            'padded_loss': 0.0,
            'consistency_loss': 0.0,
            'loss_stage_activity': 0.0
        }
        train_f1_activity = []
        train_f1_param_activity = []
        train_f1_stage_activity = []
        total_tf_count = 0.0

        for batch_idx, (jaw_id, feats, transforms, cumulative_transforms, activity, param_activity, cumulative_activity, cumulative_param_activity, directions, num_stages) in enumerate(train_loader):
            feats, transforms, cumulative_transforms, activity, param_activity, cumulative_activity, cumulative_param_activity, directions, num_stages = [
                x.to(device) if isinstance(x, torch.Tensor) else x for x in [feats, transforms, cumulative_transforms, activity, param_activity, cumulative_activity, cumulative_param_activity, directions, num_stages]
            ]
            logger.debug(f"Batch {batch_idx} shapes: feats={feats.shape}, transforms={transforms.shape}, cumulative_transforms={cumulative_transforms.shape}, directions={directions.shape}, num_stages={num_stages}")
            
            optimizer_dgcnn.zero_grad()
            optimizer_decoder.zero_grad()

            outputs = model(
                coordinates=feats,
                targets=transforms,
                cumulative_targets=cumulative_transforms,
                activity_targets=activity,
                param_activity_targets=param_activity,
                directions=directions,
                num_stages=num_stages,
                epoch=epoch,
                total_epochs=args.epochs,
                val_loss=val_loss_history['total'][-1] if val_loss_history['total'] else None,
                training=True
            )
            pred_transforms, activity_logits, param_activity_logits, stage_activity_logits, tf_count = outputs

            total_tf_count += tf_count

            total_loss, losses, f1_activity, f1_param_activity, f1_stage_activity = compute_loss(
                pred_transforms, activity_logits, param_activity_logits, stage_activity_logits,
                transforms, activity, param_activity, cumulative_transforms,
                num_stages, args.max_stages, device, logger, args
            )

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.error(f"Skipping batch {batch_idx} due to NaN/Inf in total_loss: {total_loss.item()}")
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.dgcnn.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=0.5)
            optimizer_dgcnn.step()
            optimizer_decoder.step()

            train_losses['total'] += total_loss.item()
            for key in losses:
                train_losses[key] += losses[key].item()
            train_f1_activity.append(f1_activity.item())
            train_f1_param_activity.append(f1_param_activity.item())
            train_f1_stage_activity.append(f1_stage_activity.item())

            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}/{len(train_loader)}, "
                            f"Total Loss: {total_loss.item():.4f}, MSE: {losses['loss_mse'].item():.4f}, "
                            f"Activity: {losses['loss_activity'].item():.4f}, Param Activity: {losses['loss_param_activity'].item():.4f}, "
                            f"Padded Loss: {losses['padded_loss'].item():.4f}, "
                            f"Consistency: {losses['consistency_loss'].item():.4f}, "
                            f"Stage Activity: {losses['loss_stage_activity'].item():.4f}, "
                            f"F1 Activity: {f1_activity.item():.4f}, F1 Param Activity: {f1_param_activity.item():.4f}, "
                            f"F1 Stage Activity: {f1_stage_activity.item():.4f}, TF Count: {tf_count}")

        total_possible_tf_instances = args.num_teeth * (args.max_stages - 1) * args.batch_size * len(train_loader)
        tf_percentage = (total_tf_count / total_possible_tf_instances) * 100 if total_possible_tf_instances > 0 else 0.0
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Teacher Forcing Usage: {tf_percentage:.2f}%")

        if args.use_scheduler and args.scheduler.lower() != 'reduceonplateau':
            if scheduler_dgcnn is not None:
                scheduler_dgcnn.step()
            if scheduler_decoder is not None:
                scheduler_decoder.step()

        for key in train_losses:
            train_losses[key] /= len(train_loader)
        mean_train_f1_activity = np.mean(train_f1_activity)
        mean_train_f1_param_activity = np.mean(train_f1_param_activity)
        mean_train_f1_stage_activity = np.mean(train_f1_stage_activity)

        train_loss_history['total'].append(train_losses['total'])
        train_loss_history['loss_mse'].append(train_losses['loss_mse'])
        train_loss_history['loss_activity'].append(train_losses['loss_activity'])
        train_loss_history['loss_param_activity'].append(train_losses['loss_param_activity'])
        train_loss_history['padded_loss'].append(train_losses['padded_loss'])
        train_loss_history['consistency_loss'].append(train_losses['consistency_loss'])
        train_loss_history['loss_stage_activity'].append(train_losses['loss_stage_activity'])
        train_loss_history['mean_f1_activity'].append(mean_train_f1_activity)
        train_loss_history['mean_f1_param_activity'].append(mean_train_f1_param_activity)
        train_loss_history['mean_f1_stage_activity'].append(mean_train_f1_stage_activity)

        model.eval()
        val_losses = {
            'total': 0.0,
            'loss_mse': 0.0,
            'loss_activity': 0.0,
            'loss_param_activity': 0.0,
            'padded_loss': 0.0,
            'consistency_loss': 0.0,
            'loss_stage_activity': 0.0
        }
        val_f1_activity = []
        val_f1_param_activity = []
        val_f1_stage_activity = []

        with torch.no_grad():
            for jaw_id, feats, transforms, cumulative_transforms, activity, param_activity, cumulative_activity, cumulative_param_activity, directions, num_stages in val_loader:
                feats, transforms, cumulative_transforms, activity, param_activity, cumulative_activity, cumulative_param_activity, directions, num_stages = [
                    x.to(device) if isinstance(x, torch.Tensor) else x for x in [feats, transforms, cumulative_transforms, activity, param_activity, cumulative_activity, cumulative_param_activity, directions, num_stages]
                ]

                outputs = model(
                    coordinates=feats,
                    targets=transforms,
                    cumulative_targets=cumulative_transforms,
                    activity_targets=activity,
                    param_activity_targets=param_activity,
                    directions=directions,
                    num_stages=num_stages,
                    epoch=epoch,
                    total_epochs=args.epochs,
                    val_loss=val_loss_history['total'][-1] if val_loss_history['total'] else None,
                    training=False
                )
                pred_transforms, activity_logits, param_activity_logits, stage_activity_logits,_ = outputs

                total_loss, losses, f1_activity, f1_param_activity, f1_stage_activity = compute_loss(
                    pred_transforms, activity_logits, param_activity_logits, stage_activity_logits,
                    transforms, activity, param_activity, cumulative_transforms,
                    num_stages, args.max_stages, device, logger, args
                )

                val_losses['total'] += total_loss.item()
                for key in losses:
                    val_losses[key] += losses[key].item()
                val_f1_activity.append(f1_activity.item())
                val_f1_param_activity.append(f1_param_activity.item())
                val_f1_stage_activity.append(f1_stage_activity.item())

        for key in val_losses:
            val_losses[key] /= len(val_loader)
        mean_val_f1_activity = np.mean(val_f1_activity)
        mean_val_f1_param_activity = np.mean(val_f1_param_activity)
        mean_val_f1_stage_activity = np.mean(val_f1_stage_activity)

        if args.use_scheduler and args.scheduler.lower() == 'reduceonplateau':
            if scheduler_dgcnn is not None:
                scheduler_dgcnn.step(val_losses['total'])
            if scheduler_decoder is not None:
                scheduler_decoder.step(val_losses['total'])

        logger.info(f"Epoch {epoch+1}/{args.epochs}, "
                    f"Train Loss: {train_losses['total']:.4f}, MSE: {train_losses['loss_mse']:.4f}, "
                    f"Activity: {train_losses['loss_activity']:.4f}, Param Activity: {train_losses['loss_param_activity']:.4f}, "
                    f"Padded: {train_losses['padded_loss']:.4f}, Consistency: {train_losses['consistency_loss']:.4f}, "
                    f"Stage Activity: {train_losses['loss_stage_activity']:.4f}, "
                    f"Train Mean F1 Activity: {mean_train_f1_activity:.4f}, Train Mean F1 Param Activity: {mean_train_f1_param_activity:.4f}, "
                    f"Train Mean F1 Stage Activity: {mean_train_f1_stage_activity:.4f}")
        
        logger.info(f"Epoch {epoch+1}/{args.epochs}, "
                    f"Val Loss: {val_losses['total']:.4f}, MSE: {val_losses['loss_mse']:.4f}, "
                    f"Activity: {val_losses['loss_activity']:.4f}, Param Activity: {val_losses['loss_param_activity']:.4f}, "
                    f"Padded: {val_losses['padded_loss']:.4f}, Consistency: {val_losses['consistency_loss']:.4f}, "
                    f"Stage Activity: {val_losses['loss_stage_activity']:.4f}, "
                    f"Val Mean F1 Activity: {mean_val_f1_activity:.4f}, Val Mean F1 Param Activity: {mean_val_f1_param_activity:.4f}, "
                    f"Val Mean F1 Stage Activity: {mean_val_f1_stage_activity:.4f}")

        if (epoch + 1) % 15 == 0 and epoch != 0:
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
            logger.info(f"Saved full model at epoch {epoch} with val_loss {val_losses['total']:.4f}")

            for key in train_loss_history:
                np.save(os.path.join(args.output_dir, f'train_{key}_history.npy'), np.array(train_loss_history[key]))
                logger.info(f"Saved train {key} history to {os.path.join(args.output_dir, f'train_{key}_history.npy')}")
            for key in val_loss_history:
                np.save(os.path.join(args.output_dir, f'val_{key}_history.npy'), np.array(val_loss_history[key]))
                logger.info(f"Saved validation {key} history to {os.path.join(args.output_dir, f'val_{key}_history.npy')}")

        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"No improvement in val_loss, patience counter: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
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

    for key in train_loss_history:
        np.save(os.path.join(args.output_dir, f'train_{key}_history.npy'), np.array(train_loss_history[key]))
        logger.info(f"Saved train {key} history to {os.path.join(args.output_dir, f'train_{key}_history.npy')}")
    
    for key in val_loss_history:
        np.save(os.path.join(args.output_dir, f'val_{key}_history.npy'), np.array(val_loss_history[key]))
        logger.info(f"Saved validation {key} history to {os.path.join(args.output_dir, f'val_{key}_history.npy')}")

    logger.info("Training completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Orthodontic Treatment Prediction Model")
    parser.add_argument('--data_dir', type=str, default='./Data', help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./output_decoder', help='Path to save checkpoints')
    parser.add_argument('--log_file', type=str, default='training_log_decoder.txt', help='Path to log file')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='Path to cache directory')
    parser.add_argument('--pretrained_dgcnn_path', type=str, default=None, help='Path to pretrained DGCNN weights')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint of the whole model (optional)')
    parser.add_argument('--num_points', type=int, default=4000, help='Number of points per tooth')
    parser.add_argument('--channels', type=int, default=3, help='Number of feature channels')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train/validation split ratio')
    parser.add_argument('--embed_dim', type=int, default=96, help='Embedding dimension')
    parser.add_argument('--k', type=int, default=10, help='Number of k in DGCNN')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help='MLP ratio in Transformer')
    parser.add_argument('--decoder_layers', type=int, default=1, help='Number of decoder layers in Transformer')
    parser.add_argument('--decoder_type', type=str, default='per_tooth', help='Type of used decoder model')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--teacher_forcing_prob', type=float, default=0.9, help='Teacher forcing probability')
    parser.add_argument('--max_stages', type=int, default=25, help='Maximum number of stages')
    parser.add_argument('--num_teeth', type=int, default=14, help='Number of teeth')
    parser.add_argument('--w_trans', type=float, default=1.0, help='Weight for transformation loss')
    parser.add_argument('--w_activity', type=float, default=1.0, help='Weight for activity loss')
    parser.add_argument('--w_param_activity', type=float, default=1.0, help='Weight for param activity loss')
    parser.add_argument('--w_padded', type=float, default=0.5, help='Weight for padded loss')
    parser.add_argument('--w_consistency', type=float, default=0.1, help='Weight for consistency loss')
    parser.add_argument('--w_stage_activity', type=float, default=1.0, help='Weight for stage activity loss')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'radam', 'lion', 'sparseadam', 'adan', 'caadam'], help='Optimizer type')
    parser.add_argument('--optimizer_betas', type=float, nargs=2, default=[0.9, 0.999], help='Betas for optimizer')
    parser.add_argument('--scheduler', type=str, default='cosineannealing', choices=['cosineannealing', 'reduceonplateau', 'linear'], help='Learning rate scheduler type')
    parser.add_argument('--scheduler_eta_min', type=float, default=0.0, help='Minimum learning rate for CosineAnnealing')
    parser.add_argument('--scheduler_factor', type=float, default=0.5, help='Factor for ReduceLROnPlateau')
    parser.add_argument('--scheduler_patience', type=int, default=5, help='Patience for ReduceLROnPlateau')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Number of warmup epochs')
    parser.add_argument('--warmup_start_factor', type=float, default=0.1, help='Starting factor for warmup')
    parser.add_argument('--use_scheduler', type=bool, default=False, help='Whether to use a learning rate scheduler')
    parser.add_argument('--small_error_threshold', type=float, default=0.5, help='Threshold for small errors in transformation loss')
    parser.add_argument('--small_error_scale', type=float, default=5.0, help='Scale factor for small errors in transformation loss')
    parser.add_argument('--large_delta', type=float, default=5.0, help='Delta for large errors in transformation loss')
    parser.add_argument('--sparse_weight', type=float, default=0.5, help='Weight for sparsity term in transformation loss')
    parser.add_argument('--max_error', type=float, default=40.0, help='Maximum error clamp in transformation loss')
    parser.add_argument('--seq_penalty', type=float, default=0.1, help='Penalty for non-sequential stages in stage activity loss')
    parser.add_argument('--use_focal_loss', type=bool, default=True, help='Use focal loss for activity losses')
    parser.add_argument('--focal_alpha', type=float, default=0.25, help='Alpha parameter for focal loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma parameter for focal loss')
    parser.add_argument('--use_scaler', type=bool, default=False, help='Whether to apply scaler to transformations')
    parser.add_argument('--scaler_type', type=str, default='robust', choices=['robust', 'standard'], help='Type of scaler (robust or standard)')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    train(args)