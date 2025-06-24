import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import argparse
from dataset import JawTeethDataset
from models.OrthoDGCNN_decoder import OrthoDGCNNModel
from losses_decoder import compute_loss
from optimizers import Optimizers, LRSchedulers

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
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        k=args.k,
        decoder_type=args.decoder_type,
        encoder_type=args.encoder_type
    ).to(device)

    optimizer_dgcnn = Optimizers(
        optimizer_name=args.optimizer,
        parameters=model.encoder.parameters(),
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
            if 'optimizer_dgcnn_state_dict' in checkpoint:
                optimizer_dgcnn.load_state_dict(checkpoint['optimizer_dgcnn_state_dict'])
                logger.info("Loaded optimizer_dgcnn state from checkpoint")
            if 'optimizer_decoder_state_dict' in checkpoint:
                optimizer_decoder.load_state_dict(checkpoint['optimizer_decoder_state_dict'])
                logger.info("Loaded optimizer_decoder state from checkpoint")
        except Exception as e:
            logger.error(f"Failed to load model weights from {args.checkpoint_path}: {e}")
            raise
    elif args.pretrained_dgcnn_path and os.path.exists(args.pretrained_dgcnn_path):
        checkpoint = torch.load(args.pretrained_dgcnn_path, map_location=device)
        model.encoder.load_state_dict(checkpoint['model_state_dict'])
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
            if 'scheduler_dgcnn_state_dict' in checkpoint and scheduler_dgcnn:
                scheduler_dgcnn.load_state_dict(checkpoint['scheduler_dgcnn_state_dict'])
                logger.info("Loaded scheduler_dgcnn state from checkpoint")
            if 'scheduler_decoder_state_dict' in checkpoint and scheduler_decoder:
                scheduler_decoder.load_state_dict(checkpoint['scheduler_decoder_state_dict'])
                logger.info("Loaded scheduler_decoder state from checkpoint")
        except Exception as e:
            logger.warning(f"Failed to load scheduler states from {args.checkpoint_path}: {e}")

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    train_loss_history = {
        'total': [],
        'loss_trans': [],
        'loss_padded': [],
        'loss_consistency': [],
        'loss_directions': [],
        'loss_directions_f1': [],
    }
    val_loss_history = {
        'total': [],
        'loss_trans': [],
        'loss_padded': [],
        'loss_consistency': [],
        'loss_directions': [],
        'loss_directions_f1': [],
    }

    for epoch in range(args.epochs):
        model.train()
        train_losses = {
            'total': 0.0,
            'loss_trans': 0.0,
            'loss_padded': 0.0,
            'loss_consistency': 0.0,
            'loss_directions': 0.0,
            'loss_directions_f1': 0.0,
        }

        for batch_idx, (jaw_id, feats, ratios, cumulative_transformations, directions, num_stages) in enumerate(train_loader):
            feats, ratios, cumulative_transformations, directions, num_stages = [
                x.to(device) for x in [feats, ratios, cumulative_transformations, directions, num_stages]
            ]
            logger.debug(f"Batch {batch_idx} shapes: feats={feats.shape}, ratios={ratios.shape}, cumulative_transformations={cumulative_transformations.shape}, directions={directions.shape}, num_stages={num_stages.shape}")

            optimizer_dgcnn.zero_grad()
            optimizer_decoder.zero_grad()

            outputs = model(
                coordinates=feats,
                cumulative_targets=cumulative_transformations,
                num_stages=num_stages,
                targets=ratios,
                directions=directions,
                training=True,
                epoch=epoch,
                total_epochs=args.epochs,
                val_loss=val_loss_history['total'][-1] if val_loss_history['total'] else None,
            )
            ratios_sequence, directions_sequence = outputs

            total_loss, losses = compute_loss(
                ratios_sequence=ratios_sequence,
                directions_sequence=directions_sequence,
                ratios=ratios,
                directions=directions,
                num_stages=num_stages,
                device=device,
                args=args
            )

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.error(f"Skipping batch {batch_idx} due to NaN/Inf in total_loss: {total_loss.item()}")
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=0.5)
            optimizer_dgcnn.step()
            optimizer_decoder.step()

            train_losses['total'] += total_loss.item()
            for key in losses:
                train_losses[key] += losses[key].item()

            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}/{len(train_loader)}, "
                            f"Total Loss: {total_loss.item():.4f}, Trans: {losses['loss_trans'].item():.4f}, "
                            f"Padded: {losses['loss_padded'].item():.4f}, Consistency: {losses['loss_consistency'].item():.4f}, "
                            f"Directions: {losses['loss_directions'].item():.4f}, Directions F1: {losses['loss_directions_f1'].item():.4f},")
        if args.use_scheduler and args.scheduler.lower() != 'reduceonplateau':
            if scheduler_dgcnn:
                scheduler_dgcnn.step()
            if scheduler_decoder:
                scheduler_decoder.step()

        for key in train_losses:
            train_losses[key] /= len(train_loader)

        train_loss_history['total'].append(train_losses['total'])
        for key in losses:
            train_loss_history[key].append(train_losses[key])

        model.eval()
        val_losses = {
            'total': 0.0,
            'loss_trans': 0.0,
            'loss_padded': 0.0,
            'loss_consistency': 0.0,
            'loss_directions': 0.0,
            'loss_directions_f1': 0.0,
        }

        with torch.no_grad():
            for jaw_id, feats, ratios, cumulative_transformations, directions, num_stages in val_loader:
                feats, ratios, cumulative_transformations, directions, num_stages = [
                    x.to(device) for x in [feats, ratios, cumulative_transformations, directions, num_stages]
                ]

                outputs = model(
                    coordinates=feats,
                    cumulative_targets=cumulative_transformations,
                    num_stages=num_stages,
                    targets=ratios,
                    directions=directions,
                    training=False,
                    epoch=epoch,
                    total_epochs=args.epochs,
                    val_loss=best_val_loss if best_val_loss != float('inf') else None
                )
                ratios_sequence, directions_sequence = outputs

                total_loss, losses = compute_loss(
                    ratios_sequence=ratios_sequence,
                    directions_sequence=directions_sequence,
                    ratios=ratios,
                    directions=directions,
                    num_stages=num_stages,
                    device=device,
                    args=args
                )

                val_losses['total'] += total_loss.item()
                for key in losses:
                    val_losses[key] += losses[key].item()

        for key in val_losses:
            val_losses[key] /= len(val_loader)
        
        val_loss_history['total'].append(val_losses['total'])
        for key in losses:
            val_loss_history[key].append(val_losses[key])        

        if args.use_scheduler and args.scheduler.lower() == 'reduceonplateau':
            if scheduler_dgcnn:
                scheduler_dgcnn.step(val_losses['total'])
            if scheduler_decoder:
                scheduler_decoder.step(val_losses['total'])

        logger.info(f"Epoch {epoch+1}/{args.epochs}, "
                    f"Train Loss: {train_losses['total']:.4f}, Trans: {train_losses['loss_trans']:.4f}, "
                    f"Padded: {train_losses['loss_padded']:.4f}, Consistency: {train_losses['loss_consistency']:.4f}, "
                    f"Directions: {train_losses['loss_directions']:.4f}, Directions F1: {train_losses['loss_directions_f1']:.4f},")        
        logger.info(f"Epoch {epoch+1}/{args.epochs}, "
                    f"Val Loss: {val_losses['total']:.4f}, Trans: {val_losses['loss_trans']:.4f}, "
                    f"Padded: {val_losses['loss_padded']:.4f}, Consistency: {val_losses['loss_consistency']:.4f}, "
                    f"Directions: {val_losses['loss_directions']:.4f}, Directions F1: {val_losses['loss_directions_f1']:.4f},")
        
        if (epoch + 1) % 5 == 0 and epoch != 0:
            checkpoint = {
                'epoch': epoch + 1,
                'dgcnn_state_dict': model.encoder.state_dict(),
                'decoder_state_dict': model.decoder.state_dict(),
                'ortho_dgcnn_state_dict': model.state_dict(),
                'optimizer_dgcnn_state_dict': optimizer_dgcnn.state_dict(),
                'optimizer_decoder_state_dict': optimizer_decoder.state_dict(),
                'scheduler_dgcnn_state_dict': scheduler_dgcnn.state_dict() if args.use_scheduler and scheduler_dgcnn else None,
                'scheduler_decoder_state_dict': scheduler_decoder.state_dict() if args.use_scheduler and scheduler_decoder else None,
                'val_loss': val_losses['total']
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f'model_epoch_{epoch + 1}.pth'))
            logger.info(f"Saved full model at epoch {epoch} with val_loss {val_losses['total']:.4f}")

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
                'epoch': best_epoch,
                'dgcnn_state_dict': model.encoder.state_dict(),
                'decoder_state_dict': model.decoder.state_dict(),
                'ortho_dgcnn_state_dict': model.state_dict(),
                'optimizer_dgcnn_state_dict': optimizer_dgcnn.state_dict(),
                'optimizer_decoder_state_dict': optimizer_decoder.state_dict(),
                'scheduler_dgcnn_state_dict': scheduler_dgcnn.state_dict() if args.use_scheduler and scheduler_dgcnn else None,
                'scheduler_decoder_state_dict': scheduler_decoder.state_dict() if args.use_scheduler and scheduler_decoder else None,
                'val_loss': best_val_loss
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f'best_model.pth'))
            logger.info(f"Saved best full model at epoch {best_epoch} with val_loss {best_val_loss:.4f}")
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            for key in train_loss_history:
                np.save(os.path.join(args.output_dir, f'train_{key}_history.npy'), np.array(train_loss_history[key]))
                logger.info(f"Saved train {key} history")
            for key in val_loss_history:
                np.save(os.path.join(args.output_dir, f'val_{key}_history.npy'), np.array(val_loss_history[key]))
                logger.info(f"Saved validation {key} history")

            break

        if epoch == args.epochs - 1:
            last_val_loss = val_losses['total']
            checkpoint = {
                'epoch': epoch + 1,
                'dgcnn_state_dict': model.encoder.state_dict(),
                'decoder_state_dict': model.decoder.state_dict(),
                'ortho_dgcnn_state_dict': model.state_dict(),
                'optimizer_dgcnn_state_dict': optimizer_dgcnn.state_dict(),
                'optimizer_decoder_state_dict': optimizer_decoder.state_dict(),
                'scheduler_dgcnn_state_dict': scheduler_dgcnn.state_dict() if args.use_scheduler and scheduler_dgcnn else None,
                'scheduler_decoder_state_dict': scheduler_decoder.state_dict() if args.use_scheduler and scheduler_decoder else None,
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
    parser = argparse.ArgumentParser(description="Train Orthodontic Treatment Prediction Model")
    parser.add_argument('--data_dir', type=str, default='./Data', help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./output_decoder', help='Path to save checkpoints')
    parser.add_argument('--log_file', type=str, default='training_log_decoder.txt', help='Path to log file')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='Path to cache directory')
    parser.add_argument('--pretrained_dgcnn_path', type=str, default=None, help='Path to pretrained DGCNN weights')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint of the whole model')
    parser.add_argument('--num_points', type=int, default=1000, help='Number of points per tooth')
    parser.add_argument('--channels', type=int, default=3, help='Number of feature channels')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train/validation split ratio')
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--k', type=int, default=10, help='Number of k in DGCNN')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help='MLP ratio in Transformer')
    parser.add_argument('--decoder_layers', type=int, default=1, help='Number of decoder layers')
    parser.add_argument('--decoder_type', type=str, default='per_tooth', help='Decoder type (per_tooth or transformer)')
    parser.add_argument('--encoder_type', type=str, default='dgcnn', help='Encoder type (DGCNN or pointnet++)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--max_stages', type=int, default=25, help='Maximum number of stages')
    parser.add_argument('--num_teeth', type=int, default=14, help='Number of teeth')
    parser.add_argument('--w_trans', type=float, default=2.0, help='Weight for transformation loss')
    parser.add_argument('--w_padded', type=float, default=1.0, help='Weight for padded loss')
    parser.add_argument('--w_consistency', type=float, default=0.1, help='Weight for consistency loss')
    parser.add_argument('--w_directions', type=float, default=1.0, help='Weight for direction loss')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'radam', 'lion', 'sparseadam', 'adan'], help='Optimizer type')
    parser.add_argument('--optimizer_betas', type=float, nargs=2, default=[0.9, 0.999], help='Betas for optimizer')
    parser.add_argument('--scheduler', type=str, default='cosineannealing', choices=['cosineannealing', 'reduceonplateau', 'linear'], help='Scheduler type')
    parser.add_argument('--scheduler_eta_min', type=float, default=0.0, help='Minimum learning rate')
    parser.add_argument('--scheduler_factor', type=float, default=0.5, help='Factor for ReduceLROnPlateau')
    parser.add_argument('--scheduler_patience', type=int, default=5, help='Patience for ReduceLROnPlateau')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Number of warmup epochs')
    parser.add_argument('--warmup_start_factor', type=float, default=0.1, help='Starting factor for warmup')
    parser.add_argument('--use_scheduler', type=bool, default=False, help='Use learning rate scheduler')
    parser.add_argument('--use_scaler', type=bool, default=False, help='Apply scaler to transformations')
    parser.add_argument('--scaler_type', type=str, default='robust', choices=['robust', 'standard'], help='Scaler type')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    train(args)