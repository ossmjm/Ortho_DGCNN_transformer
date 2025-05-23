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

def inverse_transform_transformations(transformations, scalers, logger):
    """Reverse parameter-specific RobustScaler transformation for transformations."""
    if not scalers or any(scaler is None for scaler in scalers):
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

def transform_cumulative_transformations(transformations, scalers, logger):
    """Apply parameter-specific RobustScaler transformation to cumulative transformations."""
    if not scalers or any(scaler is None for scaler in scalers):
        logger.warning("One or more cumulative scalers missing; returning transformations as is")
        return transformations
    try:
        shape = transformations.shape
        transformations_flat = transformations.reshape(-1, 6)
        transformed = np.zeros_like(transformations_flat)
        for i in range(6):
            transformed[:, i] = scalers[i].transform(transformations_flat[:, i].reshape(-1, 1)).flatten()
        return torch.tensor(transformed.reshape(shape), dtype=torch.float32)
    except Exception as e:
        logger.error(f"Failed to transform cumulative transformations: {e}")
        return transformations

def filter_transformations(transformations, stage_activity_probs, threshold=1e-4, inactive_proportion=0.9, logger=None):
    """Filter out entire stages based on stage activity probabilities and near-zero transformations."""
    abs_transforms = np.abs(transformations)
    near_zero = abs_transforms <= threshold
    num_values_per_stage = near_zero.shape[1] * near_zero.shape[2]
    proportion_near_zero = np.sum(near_zero, axis=(1, 2)) / num_values_per_stage
    active_stages = (proportion_near_zero < inactive_proportion) & (stage_activity_probs > 0.5)
    if not np.any(active_stages):
        if logger:
            logger.warning("All stages have mostly near-zero transformations or low activity probability")
        return transformations, np.array([])
    filtered_transforms = transformations[active_stages]
    active_stage_indices = np.where(active_stages)[0]
    if logger:
        logger.info(f"Filtered to {len(active_stage_indices)} active stages: {active_stage_indices + 1}")
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
        train_ratio=args.train_ratio,
        inference=True,
        cache_dir=args.cache_dir,
        log_file=args.log_file
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
        teacher_forcing_prob=0.0,
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
        for jaw_id, feats, cumulative_transforms, num_stages in data_loader:
            jaw_id = jaw_id[0]
            feats = feats.to(device)
            cumulative_transforms = transform_cumulative_transformations(
                cumulative_transforms.cpu().numpy(), cumulative_scalers, logger
            ).to(device)
            num_stages = torch.tensor([args.max_stages], device=device)

            try:
                pred_transforms, activity_logits, param_activity_logits, stage_activity_logits = model(
                    coordinates=feats,
                    targets=None,
                    cumulative_targets=cumulative_transforms,
                    activity_targets=None,
                    param_activity_targets=None,
                    num_stages=num_stages,
                    epoch=0,
                    total_epochs=1,
                    training=False
                )
                logger.info(f"Generated predictions for Jaw_ID {jaw_id}")
            except Exception as e:
                logger.error(f"Failed to generate predictions for Jaw_ID {jaw_id}: {e}")
                continue

            activity_probs = torch.sigmoid(activity_logits)
            activity_mask = (activity_probs > 0.7).float()
            param_activity_probs = torch.sigmoid(param_activity_logits)
            param_activity_mask = (param_activity_probs > 0.7).float() * activity_mask.unsqueeze(-1)
            stage_activity_probs = torch.sigmoid(stage_activity_logits).cpu().numpy()[0]

            masked_transforms_np = inverse_transform_transformations(
                pred_transforms.cpu().numpy(), scalers, logger
            )
            masked_transforms_np = masked_transforms_np * param_activity_mask.cpu().numpy()
            filtered_transforms, active_stage_indices = filter_transformations(
                masked_transforms_np[0], stage_activity_probs, threshold=1e-4, inactive_proportion=args.inactive_proportion, logger=logger
            )
            if len(active_stage_indices) == 0:
                logger.warning(f"No active stages for Jaw_ID {jaw_id} after filtering")
                continue
            filtered_transforms = torch.tensor(filtered_transforms, dtype=torch.float32)

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
    parser.add_argument('--data_dir', type=str, default='./Data', help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./output_decoder_predictions', help='Base path for output Excel files')
    parser.add_argument('--log_file', type=str, default='./output_decoder/inference_log.txt', help='Path to log file')
    parser.add_argument('--cache_dir', type=str, default='./Scaler', help='Path to cache directory')
    parser.add_argument('--checkpoint_path', type=str, default='./output_decoder/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--num_points', type=int, default=256, help='Number of points per tooth')
    parser.add_argument('--channels', type=int, default=4, help='Number of feature channels')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train/validation split ratio')
    parser.add_argument('--embed_dim', type=int, default=96, help='Embedding dimension')
    parser.add_argument('--k', type=int, default=10, help='Number of k in DGCNN')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help='MLP ratio in Transformer')
    parser.add_argument('--decoder_layers', type=int, default=1, help='Number of decoder layers in Transformer')
    parser.add_argument('--decoder_type', type=str, default='per_tooth', help='Type of used decoder model')
    parser.add_argument('--max_stages', type=int, default=25, help='Maximum number of stages')
    parser.add_argument('--num_teeth', type=int, default=14, help='Number of teeth')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of DataLoader workers')
    parser.add_argument('--inactive_proportion', type=float, default=0.9, help='Proportion of near-zero values to consider a stage inactive')

    args = parser.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    inference(args)