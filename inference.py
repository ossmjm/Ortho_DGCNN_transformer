import torch
from torch.utils.data import DataLoader
import os
import argparse
import logging
import ast
import pandas as pd
import numpy as np
import trimesh
from dataset import JawTeethDataset
from models.TransformerDecoder import MViTv2
from models.OrthoDGCNN import OrthoDGCNNModel

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

def apply_transformations(vertices, translation, rotation):
    if len(vertices) == 0:
        return vertices
    translation = translation.cpu().numpy()
    rotation = rotation.cpu().numpy()
    rot_matrix = np.eye(3)
    for axis, angle in enumerate(rotation):
        angle_rad = np.deg2rad(angle)
        if axis == 0:
            rot = np.array([
                [1, 0, 0],
                [0, np.cos(angle_rad), -np.sin(angle_rad)],
                [0, np.sin(angle_rad), np.cos(angle_rad)]
            ])
        elif axis == 1:
            rot = np.array([
                [np.cos(angle_rad), 0, np.sin(angle_rad)],
                [0, 1, 0],
                [-np.sin(angle_rad), 0, np.cos(angle_rad)]
            ])
        else:
            rot = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad), np.cos(angle_rad), 0],
                [0, 0, 1]
            ])
        rot_matrix = rot_matrix @ rot
    vertices_transformed = (rot_matrix @ vertices.T).T + translation
    return vertices_transformed

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized vertices_list and faces_list.
    Stacks jaw_ids, feats, cumulative_transforms, targets, and num_stages as tensors,
    but keeps vertices_list and faces_list as lists.
    """
    logger = logging.getLogger('InferenceLogger')
    
    # Unzip the batch
    jaw_ids = [item[0] for item in batch]
    feats = [item[1] for item in batch]
    cumulative_transforms = [item[2] for item in batch]
    targets = [item[3] for item in batch]
    vertices_list = [item[4] for item in batch]  # List of lists of numpy arrays
    faces_list = [item[5] for item in batch]    # List of lists of numpy arrays
    num_stages = [item[6] for item in batch]
    
    try:
        # Stack tensors where applicable
        feats = torch.stack(feats) if all(isinstance(f, torch.Tensor) for f in feats) else feats
        cumulative_transforms = torch.stack(cumulative_transforms) if all(isinstance(t, torch.Tensor) for t in cumulative_transforms) else cumulative_transforms
        targets = torch.stack(targets) if all(isinstance(t, torch.Tensor) for t in targets) else targets
        num_stages = torch.tensor(num_stages, dtype=torch.long)
        
        # Log shapes for debugging
        logger.debug(f"Collated batch: jaw_ids={len(jaw_ids)}, feats_shape={feats.shape if isinstance(feats, torch.Tensor) else 'list'}, "
                     f"num_stages={num_stages.shape}, vertices_list_len={len(vertices_list)}, faces_list_len={len(faces_list)}")
        
        return jaw_ids, feats, cumulative_transforms, targets, vertices_list, faces_list, num_stages
    except Exception as e:
        logger.error(f"Error in custom_collate_fn: {str(e)}")
        raise

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with OrthoDGCNN Model")
    parser.add_argument('--data_dir', type=str, default='./Data', help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for inference')
    parser.add_argument('--max_stages', type=int, default=25, help='Maximum number of stages')
    parser.add_argument('--num_teeth', type=int, default=14, help='Number of teeth')
    parser.add_argument('--num_points', type=int, default=256, help='Number of points per tooth')
    parser.add_argument('--channels', type=int, default=13, help='Number of feature channels')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train/validation split ratio')
    parser.add_argument('--log_file', type=str, default='inference_log.txt', help='Log file path')
    parser.add_argument('--model_path', type=str, default='./output/best_model.pth', help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory for STL files and Excel')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='Path to cache directory')
    parser.add_argument('--embed_dim', type=int, default=96, help='Embedding dimension')
    parser.add_argument('--depths', type=str, default='[1, 2, 11, 2]', help='Number of blocks per stage')
    parser.add_argument('--num_heads', type=str, default='[3, 3, 3, 3]', help='Number of attention heads per stage')
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help='MLP expansion ratio')
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help='Drop path rate')
    parser.add_argument('--decoder_layers', type=int, default=1, help='Number of decoder layers')
    return parser.parse_args()

def infer_model(args):
    logger = setup_logging(args.log_file)
    logger.info("Inference with the following arguments:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    torch.cuda.empty_cache()
    logger.info(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    
    FDI_numbers = [31, 32, 33, 34, 35, 36, 37, 41, 42, 43, 44, 45, 46, 47]
    depths = ast.literal_eval(args.depths)
    num_heads = ast.literal_eval(args.num_heads)
    
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
    logger.info(f"Dataset size: {len(dataset)}")
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    logger.info(f"Number of batches: {len(data_loader)}")
    
    mvit = MViTv2(
        embed_dim=args.embed_dim,
        num_teeth=args.num_teeth,
        max_stages=args.max_stages,
        depths=depths,
        num_heads=num_heads,
        mlp_ratio=args.mlp_ratio,
        drop_path_rate=args.drop_path_rate,
        decoder_layers=args.decoder_layers,
        teacher_forcing=False
    ).to(device)
    model = OrthoDGCNNModel(
        mvit=mvit,
        max_stages=args.max_stages,
        num_teeth=args.num_teeth,
        embed_dim=args.embed_dim,
        teacher_forcing=False,
        depths=depths,
        num_heads=num_heads,
        mlp_ratio=args.mlp_ratio,
        drop_path_rate=args.drop_path_rate
    ).to(device)
    
    logger.info(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['ortho_dgcnn_state_dict'])
    model.eval()
    
    os.makedirs(args.output_dir, exist_ok=True)
    transform_data = []
    cumulative_transform_data = []
    
    with torch.no_grad():
        for batch_idx, (jaw_ids, feats, _, _, vertices_list, faces_list, num_stages) in enumerate(data_loader):
            feats = feats.to(device)
            batch_size = feats.size(0) if isinstance(feats, torch.Tensor) else len(feats)
            
            if isinstance(feats, torch.Tensor) and (torch.isnan(feats).any() or torch.isinf(feats).any()):
                logger.warning(f"Batch {batch_idx+1}: feats contains nan/inf")
                continue
            
            with torch.amp.autocast('cuda'):
                transforms_sequence, _, _, cumulative_transforms = model(feats)  # [batch_size, max_stages, num_teeth, 6], [batch_size, num_teeth, 6]
            
            logger.debug(f"Batch {batch_idx+1}: transforms_sequence_shape={transforms_sequence.shape}, "
                         f"cumulative_transforms_shape={cumulative_transforms.shape}, "
                         f"cumulative_transforms_sample={cumulative_transforms[0, 0, :].cpu().numpy()}")
            
            if torch.isnan(transforms_sequence).any() or torch.isinf(transforms_sequence).any():
                logger.warning(f"Batch {batch_idx+1}: transforms_sequence contains nan/inf")
                continue
            if torch.isnan(cumulative_transforms).any() or torch.isinf(cumulative_transforms).any():
                logger.warning(f"Batch {batch_idx+1}: cumulative_transforms contains nan/inf")
                continue
            
            for b in range(batch_size):
                jaw_id = jaw_ids[b]
                true_stages = num_stages[b].item()
                jaw_output_dir = os.path.join(args.output_dir, jaw_id)
                os.makedirs(jaw_output_dir, exist_ok=True)
                
                for stage in range(true_stages):
                    for tooth_idx, fdi in enumerate(FDI_numbers):
                        translation = transforms_sequence[b, stage, tooth_idx, :3]
                        rotation = transforms_sequence[b, stage, tooth_idx, 3:]
                        vertices = vertices_list[b][tooth_idx]
                        faces = faces_list[b][tooth_idx]
                        
                        if len(vertices) == 0 or len(faces) == 0:
                            logger.warning(f"Jaw {jaw_id}, Stage {stage+1}, Tooth {fdi}: Empty vertices or faces")
                            continue
                        
#                        vertices_transformed = apply_transformations(vertices, translation, rotation)
#                       mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=faces, process=False)
#                        output_path = os.path.join(jaw_output_dir, f'stage_{stage+1}_tooth_{fdi}.stl')
#                        try:
#                            mesh.export(output_path)
#                            logger.info(f"Saved STL for Jaw {jaw_id}, Stage {stage+1}, Tooth {fdi} at {output_path}")
#                        except Exception as e:
#                            logger.error(f"Error saving STL for Jaw {jaw_id}, Stage {stage+1}, Tooth {fdi}: {e}")
                        
                        transform_data.append({
                            'Jaw_ID': jaw_id,
                            'Stage': stage + 1,
                            'Tooth_ID': fdi,
                            'Left/Right (mm)': translation[0].item(),
                            'Forward/Backward (mm)': translation[1].item(),
                            'Extrude/Intrude (mm)': translation[2].item(),
                            'Buccal/Lingual (degrees)': rotation[0].item(),
                            'Mesial/Distal (degrees)': rotation[1].item(),
                            'Rotation (degrees)': rotation[2].item()
                        })
                
                # Store cumulative transformations for this jaw
                for tooth_idx, fdi in enumerate(FDI_numbers):
                    cumulative_translation = cumulative_transforms[b, tooth_idx, :3]
                    cumulative_rotation = cumulative_transforms[b, tooth_idx, 3:]
                    cumulative_transform_data.append({
                        'Jaw_ID': jaw_id,
                        'Tooth_ID': fdi,
                        'Left/Right (mm)': cumulative_translation[0].item(),
                        'Forward/Backward (mm)': cumulative_translation[1].item(),
                        'Extrude/Intrude (mm)': cumulative_translation[2].item(),
                        'Buccal/Lingual (degrees)': cumulative_rotation[0].item(),
                        'Mesial/Distal (degrees)': cumulative_rotation[1].item(),
                        'Rotation (degrees)': cumulative_rotation[2].item()
                    })
    
    # Save per-stage transformations
    df = pd.DataFrame(transform_data)
    excel_path = os.path.join(args.output_dir, 'predicted_transformations.xlsx')
    try:
        df.to_excel(excel_path, index=False)
        logger.info(f"Saved transformation data to {excel_path}")
    except Exception as e:
        logger.error(f"Error saving Excel file: {e}")
    
    # Save cumulative transformations
    cumulative_df = pd.DataFrame(cumulative_transform_data)
    cumulative_excel_path = os.path.join(args.output_dir, 'predicted_cumulative_transformations.xlsx')
    try:
        cumulative_df.to_excel(cumulative_excel_path, index=False)
        logger.info(f"Saved cumulative transformation data to {cumulative_excel_path}")
    except Exception as e:
        logger.error(f"Error saving cumulative Excel file: {e}")
    
    logger.info("Inference completed")

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    infer_model(args)