import os
import torch
import pandas as pd
import numpy as np
import pickle
import logging
import argparse
from torch.utils.data import DataLoader
from dataset import JawTeethDataset
from models.OrthoDGCNN_decoder import OrthoDGCNNModel

def setup_logging(log_file):
    logger = logging.getLogger('InferenceLogger')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def inverse_transform_transformations(transformations, scalers, logger, use_scaler):
    """Reverse parameter-specific RobustScaler transformation for transformations if use_scaler is True."""
    if not use_scaler or not scalers or any(scaler is None for scaler in scalers):
        if not use_scaler:
            logger.info("use_scaler is False; returning transformations without inverse scaling")
        else:
            logger.warning("One or more scalers missing; returning transformations as is")
        return transformations
    try:
        shape = transformations.shape
        transformations_flat = transformations.reshape(-1, 6)
        transformed = np.zeros_like(transformations_flat)
        for i in range(6):
            transformed[:, i] = scalers[i].inverse_transform(transformations_flat[:, i].reshape(-1, 1)).flatten()
        return transformed.reshape(shape)
    except Exception as e:
        logger.error(f"Failed to inverse transform transformations: {e}")
        return transformations

def transform_cumulative_transformations(transformations, scalers, logger, use_scaler):
    """Apply parameter-specific RobustScaler transformation to cumulative transformations if use_scaler is True."""
    if not use_scaler or not scalers or any(scaler is None for scaler in scalers):
        if not use_scaler:
            logger.info("use_scaler is False; returning cumulative transformations without scaling")
        else:
            logger.warning("One or more cumulative scalers missing; returning transformations as tensor")
        return torch.tensor(transformations, dtype=torch.float32)
    try:
        shape = transformations.shape
        transformations_flat = transformations.reshape(-1, 6)
        transformed = np.zeros_like(transformations_flat)
        for i in range(6):
            transformed[:, i] = scalers[i].transform(transformations_flat[:, i].reshape(-1, 1)).flatten()
        return torch.tensor(transformed.reshape(shape), dtype=torch.float32)
    except Exception as e:
        logger.error(f"Failed to transform cumulative transformations: {e}")
        return torch.tensor(transformations, dtype=torch.float32)

def compute_real_transformations(ratios, directions, cumulative_transforms, logger):
    """
    Compute real transformation values from ratios and directions using:
    total_movement = net_movement / sum(predicted_ratios * directions)
    Apply only to non-zero cumulative transformations.
    Filter stages with absolute values < 1e-4.
    """
    B, S, T, P = ratios.shape  # [batch, stages, teeth, params]
    device = ratios.device
    real_transforms = torch.zeros_like(ratios)  # [B, S, T, P]
    active_stages_mask = torch.ones(B, S, dtype=torch.bool, device=device)  # [B, S]

    # Convert directions to ±1
    directions_binary = torch.where(torch.sigmoid(directions) > 0.5, torch.tensor(1.0, device=device), torch.tensor(-1.0, device=device))  # [B, S, T, P]

    # Mask for non-zero cumulative transformations
    non_zero_mask = (cumulative_transforms.abs() > 1e-6).float().unsqueeze(1)  # [B, 1, T, P]

    for b in range(B):
        for t in range(T):
            for p in range(P):
                if non_zero_mask[b, 0, t, p] == 0:
                    real_transforms[b, :, t, p] = 0.0
                    continue
                net_movement = cumulative_transforms[b, t, p]  # Scalar
                ratios_t_p = ratios[b, :, t, p]  # [S]
                directions_t_p = directions_binary[b, :, t, p]  # [S]
                denominator = (ratios_t_p * directions_t_p).sum()  # Scalar
                if abs(denominator) < 1e-6:
                    logger.warning(f"Zero or near-zero denominator for Jaw_ID batch {b}, Tooth {t}, Param {p}; setting transformations to zero")
                    real_transforms[b, :, t, p] = 0.0
                    continue
                total_movement = net_movement / denominator  # Scalar
                logger.debug(f"Computed total_movement={total_movement:.4f} for batch {b}, Tooth {t}, Param {p}")
                real_transforms[b, :, t, p] = ratios_t_p * directions_t_p * total_movement  # [S]

    # Apply non-zero mask
    real_transforms = real_transforms * non_zero_mask  # [B, S, T, P]

    # Filter stages where all absolute values are < 1e-4
    stage_active = (real_transforms.abs() >= 1e-4).any(dim=(2, 3))  # [B, S]
    active_stages_mask = stage_active  # [B, S]
    if not stage_active.any():
        logger.warning("All stages have transformations < 1e-4; no active stages")
        return real_transforms, torch.tensor([], dtype=torch.long, device=device)

    active_stage_indices = [torch.where(stage_active[b])[0] for b in range(B)]  # List of [S_active] per batch
    logger.info(f"Filtered to active stages: {[idx.tolist() for idx in active_stage_indices]}")
    return real_transforms, active_stages_mask

def filter_transformations(transformations, logger=None):
    """Filter out stages where fewer than 5 transformation values have absolute values >= 1e-3."""
    # Count values with |value| >= 1e-3 per stage
    significant_values = np.abs(transformations) >= 1e-3  # [stages, teeth, params]
    count_significant = np.sum(significant_values, axis=(1, 2))  # [stages]
    active_stages = count_significant >= 5  # Stages with at least 5 significant values
    if not np.any(active_stages):
        if logger:
            logger.warning("All stages have fewer than 5 transformation values with absolute values >= 1e-3")
        return transformations, np.array([])
    filtered_transforms = transformations[active_stages]
    active_stage_indices = np.where(active_stages)[0]
    if logger:
        logger.info(f"Filtered to {len(active_stage_indices)} active stages with at least 5 values >= 1e-3: {active_stage_indices + 1}")
    return filtered_transforms, active_stage_indices

def create_output_dataframe(jaw_id, transformations, active_stage_indices, fdi_to_index, logger):
    """Create DataFrame with transformations for active stages."""
    columns = [
        "Left/Right (mm)", "Forward/Backward (mm)", "Extrude/Intrude (mm)",
        "Buccal/Lingual (degrees)", "Mesial/Distal (degrees)", "Rotation (degrees)"
    ]
    data = []
    index_to_fdi = {v: k for k, v in fdi_to_index.items()}

    for stage_idx in range(transformations.shape[0]):
        for tooth_idx in range(transformations.shape[1]):
            transform = transformations[stage_idx, tooth_idx]
            row = {
                "Jaw_ID": jaw_id,
                "Stage": active_stage_indices[stage_idx] + 1,
                "Tooth_ID": index_to_fdi[tooth_idx],
                **{columns[i]: transform[i].item() for i in range(len(columns))}
            }
            data.append(row)

    df = pd.DataFrame(data)
    if df.empty:
        logger.warning(f"No active transformations for Jaw_ID {jaw_id}")
        return None

    expected_columns = ["Jaw_ID", "Stage", "Tooth_ID"] + columns
    df = df.reindex(columns=expected_columns, fill_value=0.0)
    return df

def inference(args):
    logger = setup_logging(args.log_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load parameter-specific scalers
    scalers = [None] * 6
    cumulative_scalers = [None] * 6
    if args.use_scaler:
        for i in range(6):
            scaler_file = os.path.join(args.cache_dir, f'scaler_param_{i}.pkl')
            cumulative_scaler_file = os.path.join(args.cache_dir, f'cumulative_scaler_param_{i}.pkl')
            if os.path.exists(scaler_file):
                try:
                    with open(scaler_file, 'rb') as f:
                        scalers[i] = pickle.load(f)
                    logger.info(f"Loaded scaler for parameter {i} from {scaler_file}")
                except Exception as e:
                    logger.error(f"Failed to load scaler for parameter {i}: {e}")
            else:
                logger.warning(f"No scaler found for parameter {i}")
            if os.path.exists(cumulative_scaler_file):
                try:
                    with open(cumulative_scaler_file, 'rb') as f:
                        cumulative_scalers[i] = pickle.load(f)
                    logger.info(f"Loaded cumulative scaler for parameter {i} from {cumulative_scaler_file}")
                except Exception as e:
                    logger.error(f"Failed to load cumulative scaler for parameter {i}: {e}")
            else:
                logger.warning(f"No cumulative scaler found for parameter {i}")

    dataset = JawTeethDataset(
        data_dir=args.data_dir,
        max_stages=args.max_stages,
        num_teeth=args.num_teeth,
        num_points=args.num_points,
        channels=args.channels,
        split='test',
        inference=True,
        cache_dir=args.cache_dir,
        log_file=args.log_file,
        use_scaler=args.use_scaler,
        scaler_type=args.scaler_type
    )
    if len(dataset) == 0:
        logger.error("Dataset is empty")
        raise ValueError("Dataset is empty")
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    logger.info(f"Dataset size: {len(dataset)}")

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
        decoder_type=args.decoder_type
    ).to(device)
    model.eval()

    if not os.path.exists(args.checkpoint_path):
        logger.error(f"Checkpoint path {args.checkpoint_path} does not exist")
        raise FileNotFoundError(f"Checkpoint path {args.checkpoint_path} does not exist")
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['ortho_dgcnn_state_dict'])
        logger.info(f"Loaded model weights from {args.checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        raise

    fdi_to_index = {
        "31": 0, "32": 1, "33": 2, "34": 3, "35": 4, "36": 5, "37": 6,
        "41": 7, "42": 8, "43": 9, "44": 10, "45": 11, "46": 12, "47": 13
    }

    with torch.no_grad():
        for jaw_id, feats, cumulative_transforms in data_loader:
            jaw_id = jaw_id[0]
            feats = feats.to(device)  # [B, num_teeth, num_points, channels]

            # Inverse scale cumulative_transformations if use_scaler=True
            cumulative_transforms_np = cumulative_transforms.cpu().numpy()  # [B, num_teeth, 6]
            if args.use_scaler:
                cumulative_transforms_np = inverse_transform_transformations(
                    cumulative_transforms_np, scalers, logger, args.use_scaler
                )
            cumulative_transforms = torch.tensor(cumulative_transforms_np, dtype=torch.float32, device=device)

            try:
                outputs = model(
                    coordinates=feats,
                    cumulative_targets=cumulative_transforms,
                    targets=None,
                    training=False,
                    epoch=0,
                    total_epochs=1,
                    val_loss=None
                )
                ratios_sequence, directions_sequence = outputs  # [B, max_stages, num_teeth, 6]
                logger.info(f"Generated predictions for Jaw_ID {jaw_id}")
            except Exception as e:
                logger.error(f"Failed to generate predictions for Jaw_ID {jaw_id}: {e}")
                continue

            # Compute real transformations
            real_transforms, active_stages_mask = compute_real_transformations(
                ratios_sequence, directions_sequence, cumulative_transforms, logger
            )  # [B, max_stages, num_teeth, 6], [B, max_stages]

            # Convert to numpy and select active stages
            real_transforms_np = real_transforms.cpu().numpy()[0]  # [max_stages, num_teeth, 6]
            active_stage_indices = torch.where(active_stages_mask[0])[0].cpu().numpy()  # [S_active]
            if len(active_stage_indices) == 0:
                logger.warning(f"No active stages for Jaw_ID {jaw_id} after filtering")
                continue
            filtered_transforms = real_transforms_np[active_stage_indices]  # [S_active, num_teeth, 6]

            df = create_output_dataframe(
                jaw_id, filtered_transforms, active_stage_indices, fdi_to_index, logger
            )
            if df is None:
                logger.warning(f"No data to save for Jaw_ID {jaw_id}")
                continue
            output_dir = args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{jaw_id}_predictions.xlsx")
            try:
                df.to_excel(output_file, index=False)
                logger.info(f"Saved predictions for Jaw_ID {jaw_id} to {output_file}")
            except Exception as e:
                logger.error(f"Failed to save predictions for Jaw_ID {jaw_id} to Excel: {e}")
                continue

    logger.info("Inference completed for all cases")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for Orthodontic Treatment Prediction")
    parser.add_argument('--data_dir', type=str, default='./Test', help='Path to test dataset folder')
    parser.add_argument('--output_dir', type=str, default='./output_decoder_predictions', help='Base path for output Excel files')
    parser.add_argument('--log_file', type=str, default='./output_decoder/inference_log.txt', help='Path to log file')
    parser.add_argument('--cache_dir', type=str, default='./Scaler', help='Path to cache directory')
    parser.add_argument('--checkpoint_path', type=str, default='./output_decoder/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--num_points', type=int, default=1000, help='Number of points per tooth')
    parser.add_argument('--channels', type=int, default=3, help='Number of feature channels')
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--k', type=int, default=20, help='Number of k in DGCNN')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help='MLP ratio in Transformer')
    parser.add_argument('--decoder_layers', type=int, default=1, help='Number of decoder layers in Transformer')
    parser.add_argument('--decoder_type', type=str, default='per_tooth', help='Type of used decoder model')
    parser.add_argument('--max_stages', type=int, default=25, help='Maximum number of stages')
    parser.add_argument('--num_teeth', type=int, default=14, help='Number of teeth')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of DataLoader workers')
    parser.add_argument('--use_scaler', type=bool, default=False, help='Whether to apply scaler to transformations')
    parser.add_argument('--scaler_type', type=str, default='robust', choices=['robust', 'standard'], help='Type of scaler (robust or standard)')

    args = parser.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    inference(args)