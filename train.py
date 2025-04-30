import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import argparse
from dataset import JawTeethDataset
from models.OrthoDGCNN import OrthoDGCNNModel
from losses import WeightedSmoothL1Loss, ConsistencyLoss, ZeroPredictionLoss, SparsityLoss, SmoothnessLoss, PaddedLoss, CumulativeLoss, CumulativeZeroLoss, CumulativeSparsityLoss

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

def compute_loss(transforms_sequence, activity_logits, type_logits, param_activity_logits, pred_cumulative, cumulative_activity_logits, cumulative_param_activity_logits, targets, activity_labels, type_labels, param_activity_labels, cumulative_transforms, cumulative_activity_labels, cumulative_param_activity_labels, true_num_stages, max_stages, device, logger, args):
    trans_loss_fn = WeightedSmoothL1Loss(beta=0.5, alpha=5.0, gamma=0.1).to(device)
    rot_loss_fn = WeightedSmoothL1Loss(beta=0.5, alpha=10.0, gamma=0.05).to(device)
    zero_trans_loss_fn = ZeroPredictionLoss(threshold=0.1, weight=args.w_zero).to(device)
    zero_rot_loss_fn = ZeroPredictionLoss(threshold=0.1, weight=args.w_zero).to(device)
    padded_loss_fn = PaddedLoss(weight=args.w_padded).to(device)
    sparsity_loss_fn = SparsityLoss(weight=args.w_sparsity).to(device)
    cumulative_loss_fn = CumulativeLoss(weight=args.w_cumulative).to(device)
    cumulative_zero_loss_fn = CumulativeZeroLoss(threshold=0.1, weight=args.w_cumulative_zero).to(device)
    cumulative_sparsity_loss_fn = CumulativeSparsityLoss(weight=args.w_cumulative_sparsity).to(device)
    consistency_loss_fn = ConsistencyLoss(weight=args.w_consistency).to(device)
    smoothness_loss_fn = SmoothnessLoss(weight=args.w_smoothness).to(device)
    bce = nn.BCEWithLogitsLoss(reduction='none').to(device)
    ce = nn.CrossEntropyLoss(reduction='none').to(device)
    
    transforms_sequence = torch.clamp(transforms_sequence, -5, 5)
    pred_cumulative = torch.clamp(pred_cumulative, -5, 5)
    
    logger.debug(f"Transforms sequence min: {transforms_sequence.min().item():.4f}, max: {transforms_sequence.max().item():.4f}, has_nan: {torch.isnan(transforms_sequence).any().item()}")
    logger.debug(f"Pred cumulative min: {pred_cumulative.min().item():.4f}, max: {pred_cumulative.max().item():.4f}, has_nan: {torch.isnan(pred_cumulative).any().item()}")
    
    pred_trans = transforms_sequence[:, :, :, :3]
    pred_rot = transforms_sequence[:, :, :, 3:]
    target_trans = targets[:, :, :, :3]
    target_rot = targets[:, :, :, 3:]
    trans_activity = param_activity_labels[:, :, :, :3]
    rot_activity = param_activity_labels[:, :, :, 3:]
    
    batch_size = transforms_sequence.size(0)
    stage_weights = torch.ones(batch_size, max_stages, device=device).clone()
    for b in range(batch_size):
        stage_weights[b, true_num_stages[b]:] = 0.0
    
    loss_trans = trans_loss_fn(pred_trans, target_trans, trans_activity, stage_weights)
    loss_rot = rot_loss_fn(pred_rot, target_rot, rot_activity, stage_weights)
    zero_trans_loss = zero_trans_loss_fn(pred_trans, target_trans, trans_activity, stage_weights)
    zero_rot_loss = zero_rot_loss_fn(pred_rot, target_rot, rot_activity, stage_weights)
    loss_activity = bce(activity_logits, activity_labels)
    loss_activity = (loss_activity * stage_weights.unsqueeze(-1)).sum() / stage_weights.unsqueeze(-1).sum().clamp(min=1e-6)
    
    logger.debug(f"Param activity logits min: {param_activity_logits.min().item():.4f}, max: {param_activity_logits.max().item():.4f}, has_nan: {torch.isnan(param_activity_logits).any().item()}")
    logger.debug(f"Param activity labels min: {param_activity_labels.min().item():.4f}, max: {param_activity_labels.max().item():.4f}, has_nan: {torch.isnan(param_activity_labels).any().item()}")
    
    # Check activity alignment for param_activity_loss
    activity_preds = torch.sigmoid(activity_logits) > 0.5
    mismatch_mask = (activity_preds & ~activity_labels.bool()).float()
    active_mask = param_activity_labels.sum(dim=-1, keepdim=True) > 0
    active_mask = active_mask.float()
    param_activity_logits = torch.clamp(param_activity_logits, -100, 100)
    loss_param_activity = bce(param_activity_logits, param_activity_labels)
    loss_param_activity = loss_param_activity * active_mask
    # Apply mismatch mask: set loss to 0.0 for mismatched teeth
    loss_param_activity = loss_param_activity * (1 - mismatch_mask.unsqueeze(-1))
    weighted_loss = (loss_param_activity * stage_weights.unsqueeze(-1).unsqueeze(-1)).sum()
    weight_sum = (stage_weights.unsqueeze(-1).unsqueeze(-1) * active_mask * (1 - mismatch_mask.unsqueeze(-1))).sum().clamp(min=1e-6)
    loss_param_activity = weighted_loss / weight_sum if weight_sum > 0 else torch.tensor(0.0, device=device)
    logger.debug(f"Param activity mismatch count: {mismatch_mask.sum().item()}")
    
    type_logits_flat = type_logits.view(-1, 4)
    type_labels_flat = type_labels.view(-1)
    loss_type = ce(type_logits_flat, type_labels_flat)
    loss_type = (loss_type.view(batch_size, max_stages, -1) * stage_weights.unsqueeze(-1)).sum() / stage_weights.unsqueeze(-1).sum().clamp(min=1e-6)
    padded_loss = padded_loss_fn(transforms_sequence, true_num_stages, max_stages)
    sparsity_loss = sparsity_loss_fn(transforms_sequence, param_activity_labels)
    consistency_loss = consistency_loss_fn(transforms_sequence, cumulative_transforms, true_num_stages, max_stages, device)
    smoothness_loss = smoothness_loss_fn(transforms_sequence)
    
    loss_cumulative = cumulative_loss_fn(pred_cumulative, cumulative_transforms)
    loss_cumulative_zero = cumulative_zero_loss_fn(pred_cumulative, cumulative_transforms, cumulative_param_activity_labels)
    loss_cumulative_sparsity = cumulative_sparsity_loss_fn(pred_cumulative, cumulative_param_activity_labels)
    loss_cumulative_activity = bce(cumulative_activity_logits, cumulative_activity_labels)
    loss_cumulative_activity = loss_cumulative_activity.mean()
    
    logger.debug(f"Cumulative param activity logits min: {cumulative_param_activity_logits.min().item():.4f}, max: {cumulative_param_activity_logits.max().item():.4f}, has_nan: {torch.isnan(cumulative_param_activity_logits).any().item()}")
    logger.debug(f"Cumulative param activity labels min: {cumulative_param_activity_labels.min().item():.4f}, max: {cumulative_param_activity_labels.max().item():.4f}, has_nan: {torch.isnan(cumulative_param_activity_labels).any().item()}")
    
    # Check activity alignment for cumulative_param_activity_loss
    cumulative_activity_preds = torch.sigmoid(cumulative_activity_logits) > 0.5
    cumulative_mismatch_mask = (cumulative_activity_preds & ~cumulative_activity_labels.bool()).float()
    cumulative_active_mask = cumulative_param_activity_labels.sum(dim=-1, keepdim=True) > 0
    cumulative_active_mask = cumulative_active_mask.float()
    cumulative_param_activity_logits = torch.clamp(cumulative_param_activity_logits, -100, 100)
    loss_cumulative_param_activity = bce(cumulative_param_activity_logits, cumulative_param_activity_labels)
    loss_cumulative_param_activity = loss_cumulative_param_activity * cumulative_active_mask
    # Apply mismatch mask
    loss_cumulative_param_activity = loss_cumulative_param_activity * (1 - cumulative_mismatch_mask.unsqueeze(-1))
    weight_sum_cumulative = (cumulative_active_mask * (1 - cumulative_mismatch_mask.unsqueeze(-1))).sum().clamp(min=1e-6)
    loss_cumulative_param_activity = loss_cumulative_param_activity.sum() / weight_sum_cumulative if weight_sum_cumulative > 0 else torch.tensor(0.0, device=device)
    logger.debug(f"Cumulative param activity mismatch count: {cumulative_mismatch_mask.sum().item()}")
    
    losses = {
        'loss_trans': loss_trans, 'loss_rot': loss_rot, 'zero_trans_loss': zero_trans_loss, 'zero_rot_loss': zero_rot_loss,
        'loss_activity': loss_activity, 'loss_param_activity': loss_param_activity, 'loss_type': loss_type,
        'padded_loss': padded_loss, 'sparsity_loss': sparsity_loss, 'consistency_loss': consistency_loss, 'smoothness_loss': smoothness_loss,
        'loss_cumulative': loss_cumulative, 'loss_cumulative_zero': loss_cumulative_zero, 'loss_cumulative_sparsity': loss_cumulative_sparsity,
        'loss_cumulative_activity': loss_cumulative_activity, 'loss_cumulative_param_activity': loss_cumulative_param_activity
    }
    for name, loss in losses.items():
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"{name} is NaN or Inf: {loss.item()}")
    
    stagewise_loss = (
        args.w_trans * loss_trans +
        args.w_rot * loss_rot +
        args.w_zero * (zero_trans_loss + zero_rot_loss) +
        args.w_activity * loss_activity +
        args.w_param_activity * loss_param_activity +
        args.w_type * loss_type +
        args.w_padded * padded_loss +
        args.w_sparsity * sparsity_loss +
        args.w_consistency * consistency_loss +
        args.w_smoothness * smoothness_loss
    )
    
    cumulative_loss = (
        args.w_cumulative * loss_cumulative +
        args.w_cumulative_zero * loss_cumulative_zero +
        args.w_cumulative_sparsity * loss_cumulative_sparsity +
        args.w_cumulative_activity * loss_cumulative_activity +
        args.w_cumulative_param_activity * loss_cumulative_param_activity
    )
    
    total_loss = stagewise_loss + cumulative_loss
    
    if torch.isnan(stagewise_loss) or torch.isinf(stagewise_loss):
        logger.error(f"Stagewise loss is NaN or Inf: {stagewise_loss.item()}")
    if torch.isnan(cumulative_loss) or torch.isinf(cumulative_loss):
        logger.error(f"Cumulative loss is NaN or Inf: {cumulative_loss.item()}")
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        logger.error(f"Total loss is NaN or Inf: {total_loss.item()}")
    
    tooth_errors = torch.mean(torch.abs(transforms_sequence - targets) * activity_labels.unsqueeze(-1), dim=(0, 1, 3))
    for tooth_idx in range(args.num_teeth):
        logger.debug(f"Tooth {tooth_idx+31}: Mean Absolute Error = {tooth_errors[tooth_idx]:.4f}")
    
    activity_preds = (torch.sigmoid(activity_logits) > 0.5).float()
    activity_accuracy = (activity_preds == activity_labels).float().mean()
    type_preds = torch.argmax(type_logits, dim=-1)
    type_accuracy = (type_preds == type_labels).float().mean()
    param_activity_preds = (torch.sigmoid(param_activity_logits) > 0.5).float()
    param_activity_accuracy = (param_activity_preds == param_activity_labels).float().mean()
    cumulative_activity_preds = (torch.sigmoid(cumulative_activity_logits) > 0.5).float()
    cumulative_activity_accuracy = (cumulative_activity_preds == cumulative_activity_labels).float().mean()
    cumulative_param_activity_preds = (torch.sigmoid(cumulative_param_activity_logits) > 0.5).float()
    cumulative_param_activity_accuracy = (cumulative_param_activity_preds == cumulative_param_activity_labels).float().mean()
    logger.debug(f"Activity prediction accuracy: {activity_accuracy:.4f}")
    logger.debug(f"Type prediction accuracy: {type_accuracy:.4f}")
    logger.debug(f"Param activity prediction accuracy: {param_activity_accuracy:.4f}")
    logger.debug(f"Cumulative activity prediction accuracy: {cumulative_activity_accuracy:.4f}")
    logger.debug(f"Cumulative param activity prediction accuracy: {cumulative_param_activity_accuracy:.4f}")
    
    zero_trans_pred = (torch.abs(pred_trans) < 0.01).float().mean()
    zero_trans_target = (target_trans == 0).float().mean()
    zero_rot_pred = (torch.abs(pred_rot) < 0.01).float().mean()
    zero_rot_target = (target_rot == 0).float().mean()
    zero_cumulative_pred = (torch.abs(pred_cumulative) < 0.01).float().mean()
    zero_cumulative_target = (cumulative_transforms == 0).float().mean()
    logger.debug(f"Zero Trans Pred: {zero_trans_pred:.4f}, Target: {zero_trans_target:.4f}")
    logger.debug(f"Zero Rot Pred: {zero_rot_pred:.4f}, Target: {zero_rot_target:.4f}")
    logger.debug(f"Zero Cumulative Pred: {zero_cumulative_pred:.4f}, Target: {zero_cumulative_target:.4f}")
    
    return (
        total_loss, stagewise_loss, cumulative_loss,
        loss_trans, loss_rot, zero_trans_loss, zero_rot_loss, loss_activity, loss_param_activity, loss_type,
        padded_loss, sparsity_loss, loss_cumulative, loss_cumulative_zero, loss_cumulative_sparsity,
        loss_cumulative_activity, loss_cumulative_param_activity, consistency_loss, smoothness_loss
    )

def check_gradients(model, logger, stage):
    for name, param in model.named_parameters():
        if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
            logger.error(f"NaN or Inf gradient in {name} at stage {stage}, grad min: {param.grad.min().item():.4f}, max: {param.grad.max().item():.4f}")
# ... (imports and other functions remain unchanged)

def train(args):
    logger = setup_logging(args.log_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    logger.info(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

    model = OrthoDGCNNModel(
        max_stages=args.max_stages,
        num_teeth=args.num_teeth,
        embed_dim=args.embed_dim,
        teacher_forcing=args.teacher_forcing_prob > 0,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        decoder_layers=args.decoder_layers,
        k=args.k
    ).to(device)

    optimizer_dgcnn = optim.AdamW(model.dgcnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_decoder = optim.AdamW(model.decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_cumulative = optim.AdamW(model.cumulative_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler_dgcnn = optim.lr_scheduler.CosineAnnealingLR(optimizer_dgcnn, T_max=args.epochs)
    scheduler_decoder = optim.lr_scheduler.CosineAnnealingLR(optimizer_decoder, T_max=args.epochs)
    scheduler_cumulative = optim.lr_scheduler.CosineAnnealingLR(optimizer_cumulative, T_max=args.epochs)
    
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_losses = {
            'total': 0.0, 'stagewise': 0.0, 'cumulative': 0.0,
            'transform_translation': 0.0, 'transform_rotation': 0.0, 'zero_trans': 0.0, 'zero_rot': 0.0,
            'activity': 0.0, 'param_activity': 0.0, 'type': 0.0, 'padded': 0.0, 'sparsity': 0.0,
            'cumulative_loss': 0.0, 'cumulative_zero': 0.0, 'cumulative_sparsity': 0.0,
            'cumulative_activity': 0.0, 'cumulative_param_activity': 0.0, 'consistency': 0.0, 'smoothness': 0.0
        }
        for batch_idx, (jaw_id, feats, transforms, cumulative_transforms, activity, param_activity, type_labels, cumulative_activity, cumulative_param_activity, num_stages) in enumerate(train_loader):
            feats, transforms, cumulative_transforms, activity, param_activity, type_labels, cumulative_activity, cumulative_param_activity, num_stages = [
                x.to(device) for x in [feats, transforms, cumulative_transforms, activity, param_activity, type_labels, cumulative_activity, cumulative_param_activity, num_stages]
            ]
            
            logger.debug(f"Batch {batch_idx} shapes: feats={feats.shape}, transforms={transforms.shape}, "
                        f"cumulative_transforms={cumulative_transforms.shape}, activity={activity.shape}, "
                        f"param_activity={param_activity.shape}, type_labels={type_labels.shape}, "
                        f"cumulative_activity={cumulative_activity.shape}, "
                        f"cumulative_param_activity={cumulative_param_activity.shape}, num_stages={num_stages.shape}")
            logger.debug(f"Cumulative transforms input min: {cumulative_transforms.min().item():.4f}, max: {cumulative_transforms.max().item():.4f}, has_nan: {torch.isnan(cumulative_transforms).any().item()}")

            optimizer_dgcnn.zero_grad()
            optimizer_decoder.zero_grad()
            optimizer_cumulative.zero_grad()

            pred_transforms, activity_logits, type_logits, param_activity_logits, pred_cumulative, cumulative_activity_logits, cumulative_param_activity_logits = model(
                feats, targets=transforms, cumulative_targets=cumulative_transforms, epoch=epoch, total_epochs=args.epochs
            )

            stage_weights = torch.ones(feats.size(0), args.max_stages, device=device).clone()
            for i in range(feats.size(0)):
                stage_weights[i, num_stages[i]:] = 0.0
            
            total_loss, stagewise_loss, cumulative_loss, loss_trans, loss_rot, zero_trans_loss, zero_rot_loss, loss_activity, loss_param_activity, loss_type, padded_loss, sparsity_loss, loss_cumulative, loss_cumulative_zero, loss_cumulative_sparsity, loss_cumulative_activity, loss_cumulative_param_activity, consistency_loss, smoothness_loss = compute_loss(
                pred_transforms, activity_logits, type_logits, param_activity_logits, pred_cumulative, cumulative_activity_logits, cumulative_param_activity_logits,
                transforms, activity, type_labels, param_activity, cumulative_transforms, cumulative_activity, cumulative_param_activity, num_stages, args.max_stages, device, logger, args
            )

            check_gradients(model.decoder, logger, "before stagewise_loss")
            stagewise_loss.backward(retain_graph=True)
            check_gradients(model.decoder, logger, "after stagewise_loss")
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=0.5)
            optimizer_decoder.step()
            optimizer_decoder.zero_grad()
            optimizer_dgcnn.zero_grad()  # Clear dgcnn gradients after decoder update

            check_gradients(model.cumulative_model, logger, "before cumulative_loss")
            cumulative_loss.backward(retain_graph=True)
            check_gradients(model.cumulative_model, logger, "after cumulative_loss")
            torch.nn.utils.clip_grad_norm_(model.cumulative_model.parameters(), max_norm=0.5)
            optimizer_cumulative.step()
            optimizer_cumulative.zero_grad()
            optimizer_dgcnn.zero_grad()  # Clear dgcnn gradients after cumulative update

            check_gradients(model.dgcnn, logger, "before total_loss")
            total_loss.backward()
            check_gradients(model.dgcnn, logger, "after total_loss")
            torch.nn.utils.clip_grad_norm_(model.dgcnn.parameters(), max_norm=0.5)
            optimizer_dgcnn.step()
            optimizer_dgcnn.zero_grad()
            optimizer_decoder.zero_grad()
            optimizer_cumulative.zero_grad()

            scheduler_dgcnn.step()
            scheduler_decoder.step()
            scheduler_cumulative.step()

            train_losses['total'] += total_loss.item()
            train_losses['stagewise'] += stagewise_loss.item()
            train_losses['cumulative'] += cumulative_loss.item()
            train_losses['transform_translation'] += loss_trans.item()
            train_losses['transform_rotation'] += loss_rot.item()
            train_losses['zero_trans'] += zero_trans_loss.item()
            train_losses['zero_rot'] += zero_rot_loss.item()
            train_losses['activity'] += loss_activity.item()
            train_losses['param_activity'] += loss_param_activity.item()
            train_losses['type'] += loss_type.item()
            train_losses['padded'] += padded_loss.item()
            train_losses['sparsity'] += sparsity_loss.item()
            train_losses['cumulative_loss'] += loss_cumulative.item()
            train_losses['cumulative_zero'] += loss_cumulative_zero.item()
            train_losses['cumulative_sparsity'] += loss_cumulative_sparsity.item()
            train_losses['cumulative_activity'] += loss_cumulative_activity.item()
            train_losses['cumulative_param_activity'] += loss_cumulative_param_activity.item()
            train_losses['consistency'] += consistency_loss.item()
            train_losses['smoothness'] += smoothness_loss.item()

            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}/{len(train_loader)}, "
                            f"Total Loss: {total_loss.item():.4f}, Stagewise Loss: {stagewise_loss.item():.4f}, "
                            f"Cumulative Loss: {cumulative_loss.item():.4f}, Transform_Translation: {loss_trans.item():.4f}, "
                            f"Transform_Rotation: {loss_rot.item():.4f}, Zero_Trans: {zero_trans_loss.item():.4f}, "
                            f"Zero_Rot: {zero_rot_loss.item():.4f}, Activity: {loss_activity.item():.4f}, "
                            f"Param_Activity: {loss_param_activity.item():.4f}, Type: {loss_type.item():.4f}, "
                            f"Padded: {padded_loss.item():.4f}, Sparsity: {sparsity_loss.item():.4f}, "
                            f"Cumulative: {loss_cumulative.item():.4f}, Cumulative_Zero: {loss_cumulative_zero.item():.4f}, "
                            f"Cumulative_Sparsity: {loss_cumulative_sparsity.item():.4f}, "
                            f"Cumulative_Activity: {loss_cumulative_activity.item():.4f}, "
                            f"Cumulative_Param_Activity: {loss_cumulative_param_activity.item():.4f}, "
                            f"Consistency: {consistency_loss.item():.4f}, Smoothness: {loss_smoothness.item():.4f}")

        for key in train_losses:
            train_losses[key] /= len(train_loader)

        model.eval()
        val_losses = {
            'total': 0.0, 'stagewise': 0.0, 'cumulative': 0.0,
            'transform_translation': 0.0, 'transform_rotation': 0.0, 'zero_trans': 0.0, 'zero_rot': 0.0,
            'activity': 0.0, 'param_activity': 0.0, 'type': 0.0, 'padded': 0.0, 'sparsity': 0.0,
            'cumulative_loss': 0.0, 'cumulative_zero': 0.0, 'cumulative_sparsity': 0.0,
            'cumulative_activity': 0.0, 'cumulative_param_activity': 0.0, 'consistency': 0.0, 'smoothness': 0.0
        }
        with torch.no_grad():
            for jaw_id, feats, transforms, cumulative_transforms, activity, param_activity, type_labels, cumulative_activity, cumulative_param_activity, num_stages in val_loader:
                feats, transforms, cumulative_transforms, activity, param_activity, type_labels, cumulative_activity, cumulative_param_activity, num_stages = [
                    x.to(device) for x in [feats, transforms, cumulative_transforms, activity, param_activity, type_labels, cumulative_activity, cumulative_param_activity, num_stages]
                ]

                pred_transforms, activity_logits, type_logits, param_activity_logits, pred_cumulative, cumulative_activity_logits, cumulative_param_activity_logits = model(
                    feats, targets=transforms, cumulative_targets=cumulative_transforms, epoch=epoch, total_epochs=args.epochs
                )

                stage_weights = torch.ones(feats.size(0), args.max_stages, device=device).clone()
                for i in range(feats.size(0)):
                    stage_weights[i, num_stages[i]:] = 0.0

                total_loss, stagewise_loss, cumulative_loss, loss_trans, loss_rot, zero_trans_loss, zero_rot_loss, loss_activity, loss_param_activity, loss_type, padded_loss, sparsity_loss, loss_cumulative, loss_cumulative_zero, loss_cumulative_sparsity, loss_cumulative_activity, loss_cumulative_param_activity, consistency_loss, smoothness_loss = compute_loss(
                    pred_transforms, activity_logits, type_logits, param_activity_logits, pred_cumulative, cumulative_activity_logits, cumulative_param_activity_logits,
                    transforms, activity, type_labels, param_activity, cumulative_transforms, cumulative_activity, cumulative_param_activity, num_stages, args.max_stages, device, logger, args
                )

                val_losses['total'] += total_loss.item()
                val_losses['stagewise'] += stagewise_loss.item()
                val_losses['cumulative'] += cumulative_loss.item()
                val_losses['transform_translation'] += loss_trans.item()
                val_losses['transform_rotation'] += loss_rot.item()
                val_losses['zero_trans'] += zero_trans_loss.item()
                val_losses['zero_rot'] += zero_rot_loss.item()
                val_losses['activity'] += loss_activity.item()
                val_losses['param_activity'] += loss_param_activity.item()
                val_losses['type'] += loss_type.item()
                val_losses['padded'] += padded_loss.item()
                val_losses['sparsity'] += sparsity_loss.item()
                val_losses['cumulative_loss'] += loss_cumulative.item()
                val_losses['cumulative_zero'] += loss_cumulative_zero.item()
                val_losses['cumulative_sparsity'] += loss_cumulative_sparsity.item()
                val_losses['cumulative_activity'] += loss_cumulative_activity.item()
                val_losses['cumulative_param_activity'] += loss_cumulative_param_activity.item()
                val_losses['consistency'] += consistency_loss.item()
                val_losses['smoothness'] += smoothness_loss.item()

        for key in val_losses:
            val_losses[key] /= len(val_loader)

        logger.info(f"Epoch {epoch+1}/{args.epochs}, "
                    f"Train Loss: {train_losses['total']:.4f} (Stagewise: {train_losses['stagewise']:.4f}, "
                    f"Cumulative: {train_losses['cumulative']:.4f}, Translation: {train_losses['transform_translation']:.4f}, "
                    f"Rotation: {train_losses['transform_rotation']:.4f}, Zero_Trans: {train_losses['zero_trans']:.4f}, "
                    f"Zero_Rot: {train_losses['zero_rot']:.4f}, Activity: {train_losses['activity']:.4f}, "
                    f"Param_Activity: {train_losses['param_activity']:.4f}, Type: {train_losses['type']:.4f}, "
                    f"Padded: {train_losses['padded']:.4f}, Sparsity: {train_losses['sparsity']:.4f}, "
                    f"Cumulative_Loss: {train_losses['cumulative_loss']:.4f}, Cumulative_Zero: {train_losses['cumulative_zero']:.4f}, "
                    f"Cumulative_Sparsity: {train_losses['cumulative_sparsity']:.4f}, "
                    f"Cumulative_Activity: {train_losses['cumulative_activity']:.4f}, "
                    f"Cumulative_Param_Activity: {train_losses['cumulative_param_activity']:.4f}, "
                    f"Consistency: {train_losses['consistency']:.4f}, Smoothness: {train_losses['smoothness']:.4f}), "
                    f"Val Loss: {val_losses['total']:.4f} (Stagewise: {val_losses['stagewise']:.4f}, "
                    f"Cumulative: {val_losses['cumulative']:.4f}, Translation: {val_losses['transform_translation']:.4f}, "
                    f"Rotation: {val_losses['transform_rotation']:.4f}, Zero_Trans: {val_losses['zero_trans']:.4f}, "
                    f"Zero_Rot: {val_losses['zero_rot']:.4f}, Activity: {val_losses['activity']:.4f}, "
                    f"Param_Activity: {val_losses['param_activity']:.4f}, Type: {val_losses['type']:.4f}, "
                    f"Padded: {val_losses['padded']:.4f}, Sparsity: {val_losses['sparsity']:.4f}, "
                    f"Cumulative_Loss: {val_losses['cumulative_loss']:.4f}, Cumulative_Zero: {val_losses['cumulative_zero']:.4f}, "
                    f"Cumulative_Sparsity: {val_losses['cumulative_sparsity']:.4f}, "
                    f"Cumulative_Activity: {val_losses['cumulative_activity']:.4f}, "
                    f"Cumulative_Param_Activity: {val_losses['cumulative_param_activity']:.4f}, "
                    f"Consistency: {val_losses['consistency']:.4f}, Smoothness: {val_losses['smoothness']:.4f})")

        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            patience_counter = 0
            checkpoint = {
                'epoch': epoch + 1,
                'dgcnn_state_dict': model.dgcnn.state_dict(),
                'decoder_state_dict': model.decoder.state_dict(),
                'cumulative_model_state_dict': model.cumulative_model.state_dict(),
                'ortho_dgcnn_state_dict': model.state_dict(),
                'optimizer_dgcnn_state_dict': optimizer_dgcnn.state_dict(),
                'optimizer_decoder_state_dict': optimizer_decoder.state_dict(),
                'optimizer_cumulative_state_dict': optimizer_cumulative.state_dict(),
                'val_loss': best_val_loss
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pth'))
            logger.info(f"Saved best model at epoch {epoch+1} with val_loss {best_val_loss:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    logger.info("Training completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Orthodontic Treatment Prediction Model")
    parser.add_argument('--data_dir', type=str, default='./Data', help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./output', help='Path to save checkpoints')
    parser.add_argument('--log_file', type=str, default='training_log.txt', help='Path to log file')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='Path to cache directory')
    parser.add_argument('--num_points', type=int, default=256, help='Number of points per tooth')
    parser.add_argument('--channels', type=int, default=13, help='Number of feature channels')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train/validation split ratio')
    parser.add_argument('--embed_dim', type=int, default=96, help='Embedding dimension')
    parser.add_argument('--k', type=int, default=10, help='Number of k in DGCNN')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of heads per stage')
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help='MLP ratio in MViTv2')
    parser.add_argument('--decoder_layers', type=int, default=1, help='Number of decoder layers')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--teacher_forcing_prob', type=float, default=0.5, help='Teacher forcing probability')
    parser.add_argument('--max_stages', type=int, default=25, help='Maximum number of stages')
    parser.add_argument('--num_teeth', type=int, default=14, help='Number of teeth')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--w_trans', type=float, default=1.0, help='Weight for translation loss')
    parser.add_argument('--w_rot', type=float, default=2.0, help='Weight for rotation loss')
    parser.add_argument('--w_zero', type=float, default=1.0, help='Weight for zero prediction loss')
    parser.add_argument('--w_activity', type=float, default=1.0, help='Weight for activity loss')
    parser.add_argument('--w_param_activity', type=float, default=1.0, help='Weight for param activity loss')
    parser.add_argument('--w_type', type=float, default=0.5, help='Weight for type loss')
    parser.add_argument('--w_padded', type=float, default=0.01, help='Weight for padded loss')
    parser.add_argument('--w_sparsity', type=float, default=1.0, help='Weight for sparsity loss')
    parser.add_argument('--w_cumulative', type=float, default=1.0, help='Weight for cumulative loss')
    parser.add_argument('--w_cumulative_zero', type=float, default=1.0, help='Weight for cumulative zero loss')
    parser.add_argument('--w_cumulative_sparsity', type=float, default=1.0, help='Weight for cumulative sparsity loss')
    parser.add_argument('--w_cumulative_activity', type=float, default=1.0, help='Weight for cumulative activity loss')
    parser.add_argument('--w_cumulative_param_activity', type=float, default=1.0, help='Weight for cumulative param activity loss')
    parser.add_argument('--w_consistency', type=float, default=0.01, help='Weight for consistency loss')
    parser.add_argument('--w_smoothness', type=float, default=0.01, help='Weight for smoothness loss')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    train(args)