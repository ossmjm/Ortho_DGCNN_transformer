import os
import logging
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from dataset import JawTeethDataset
from models.OrthoDGCNN import OrthoDGCNNModel
from models.DGCNN import DGCNN
from models.StageTransformer import StageTransformer
from train import compute_loss, WeightedSmoothL1Loss, setup_logging
import trimesh
from scipy.spatial.transform import Rotation

# Custom collate function to handle variable-sized vertices_list and faces_list
def custom_collate_fn(batch):
    # batch is a list of tuples from JawTeethDataset.__getitem__
    # Each tuple: (cordinates, transformations, vertices_list, faces_list, num_stages, jaw_id)
    
    cordinates = [item[0] for item in batch]
    transformations = [item[1] for item in batch]
    vertices_list = [item[2] for item in batch]  # Keep as list
    faces_list = [item[3] for item in batch]    # Keep as list
    num_stages = [item[4] for item in batch]
    jaw_ids = [item[5] for item in batch]
    
    # Use default_collate for tensors
    cordinates = default_collate(cordinates)
    transformations = default_collate(transformations)
    num_stages = default_collate(num_stages)
    
    # jaw_ids remains a list of strings
    return (cordinates, transformations, vertices_list, faces_list, num_stages, jaw_ids)

def rotation_matrix_from_euler(angles, order='xyz'):
    """
    Convert Euler angles to rotation matrix.
    
    Args:
        angles (torch.Tensor): Euler angles in degrees of shape (..., 3)
        order (str): Order of rotation axes, e.g., 'xyz'
    
    Returns:
        torch.Tensor: Rotation matrix of shape (..., 3, 3)
    """
    angles = torch.deg2rad(angles)
    c = torch.cos(angles)
    s = torch.sin(angles)
    
    if order == 'xyz':
        Rx = torch.stack([
            torch.stack([torch.ones_like(c[..., 0]), torch.zeros_like(c[..., 0]), torch.zeros_like(c[..., 0])], dim=-1),
            torch.stack([torch.zeros_like(c[..., 0]), c[..., 0], -s[..., 0]], dim=-1),
            torch.stack([torch.zeros_like(c[..., 0]), s[..., 0], c[..., 0]], dim=-1)
        ], dim=-2)
        
        Ry = torch.stack([
            torch.stack([c[..., 1], torch.zeros_like(c[..., 1]), s[..., 1]], dim=-1),
            torch.stack([torch.zeros_like(c[..., 1]), torch.ones_like(c[..., 1]), torch.zeros_like(c[..., 1])], dim=-1),
            torch.stack([-s[..., 1], torch.zeros_like(c[..., 1]), c[..., 1]], dim=-1)
        ], dim=-2)
        
        Rz = torch.stack([
            torch.stack([c[..., 2], -s[..., 2], torch.zeros_like(c[..., 2])], dim=-1),
            torch.stack([s[..., 2], c[..., 2], torch.zeros_like(c[..., 2])], dim=-1),
            torch.stack([torch.zeros_like(c[..., 2]), torch.zeros_like(c[..., 2]), torch.ones_like(c[..., 2])], dim=-1)
        ], dim=-2)
        
        R = torch.matmul(torch.matmul(Rx, Ry), Rz)
    
    return R

def apply_transformations(vertices_list, transforms_sequence, activity_logits, param_activity_logits, logger):
    """
    Apply predicted transformations to the vertices for each tooth and stage.
    
    Args:
        vertices_list (list): List of vertex arrays for each batch and tooth
        transforms_sequence (torch.Tensor): Predicted transformations of shape (batch_size, max_stages, num_teeth, 6)
        activity_logits (torch.Tensor): Activity logits of shape (batch_size, max_stages, num_teeth)
        param_activity_logits (torch.Tensor): Parameter activity logits of shape (batch_size, max_stages, num_teeth, 6)
        logger (logging.Logger): Logger for debugging
    
    Returns:
        list: List of transformed vertices for each batch, stage, and tooth
    """
    batch_size, max_stages, num_teeth, _ = transforms_sequence.shape
    activity_probs = torch.sigmoid(activity_logits) > 0.5
    param_activity_probs = torch.sigmoid(param_activity_logits) > 0.5
    
    all_stages_vertices = []
    
    for batch_idx in range(batch_size):
        batch_vertices = []
        for stage_idx in range(max_stages):
            stage_vertices = []
            for tooth_idx in range(num_teeth):
                if not activity_probs[batch_idx, stage_idx, tooth_idx]:
                    stage_vertices.append(vertices_list[batch_idx][tooth_idx])
                    continue
                
                vertices = torch.tensor(vertices_list[batch_idx][tooth_idx], dtype=torch.float32, device=transforms_sequence.device)
                trans = transforms_sequence[batch_idx, stage_idx, tooth_idx, :3]
                rot = transforms_sequence[batch_idx, stage_idx, tooth_idx, 3:]
                
                for param_idx in range(6):
                    if not param_activity_probs[batch_idx, stage_idx, tooth_idx, param_idx]:
                        if param_idx < 3:
                            trans[param_idx] = 0.0
                        else:
                            rot[param_idx - 3] = 0.0
                
                R = rotation_matrix_from_euler(rot, order='xyz')
                transformed_vertices = vertices @ R.transpose(-1, -2) + trans
                stage_vertices.append(transformed_vertices.cpu().numpy())
            
            batch_vertices.append(stage_vertices)
        all_stages_vertices.append(batch_vertices)
    
    return all_stages_vertices

def generate_stl_files(stages_vertices, faces_list, true_num_stages, output_dir, logger):
    """
    Generate STL files for each stage, tooth, and batch.
    
    Args:
        stages_vertices (list): List of transformed vertices for each batch, stage, and tooth
        faces_list (list): List of face arrays for each batch and tooth
        true_num_stages (list): List of true number of stages for each batch
        output_dir (str): Directory to save STL files
        logger (logging.Logger): Logger for debugging
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for batch_idx, (batch_vertices, batch_faces, num_stages) in enumerate(zip(stages_vertices, faces_list, true_num_stages)):
        for stage_idx in range(num_stages.item()):
            for tooth_idx in range(len(batch_vertices[stage_idx])):
                vertices = batch_vertices[stage_idx][tooth_idx]
                faces = batch_faces[tooth_idx]
                
                if len(faces) == 0 or len(vertices) == 0:
                    logger.warning(f"Batch {batch_idx+1}, Stage {stage_idx+1}, Tooth {tooth_idx+31}: Empty vertices or faces")
                    continue
                
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                output_path = os.path.join(output_dir, f'jaw_{batch_idx}_stage_{stage_idx+1}_tooth_{tooth_idx+31}.stl')
                mesh.export(output_path)
                logger.info(f"Saved STL file: {output_path}")

def save_transformations(transforms_sequence, activity_logits, param_activity_logits, output_dir, logger):
    """
    Save transformation matrices for each stage and tooth.
    
    Args:
        transforms_sequence (torch.Tensor): Predicted transformations of shape (batch_size, max_stages, num_teeth, 6)
        activity_logits (torch.Tensor): Activity logits of shape (batch_size, max_stages, num_teeth)
        param_activity_logits (torch.Tensor): Parameter activity logits of shape (batch_size, max_stages, num_teeth, 6)
        output_dir (str): Directory to save transformation files
        logger (logging.Logger): Logger for debugging
    """
    os.makedirs(output_dir, exist_ok=True)
    batch_size, max_stages, num_teeth, _ = transforms_sequence.shape
    activity_probs = torch.sigmoid(activity_logits) > 0.5
    param_activity_probs = torch.sigmoid(param_activity_logits) > 0.5
    
    for batch_idx in range(batch_size):
        for stage_idx in range(max_stages):
            output_path = os.path.join(output_dir, f'jaw_{batch_idx}_stage_{stage_idx+1}_transform.txt')
            with open(output_path, 'w') as f:
                for tooth_idx in range(num_teeth):
                    if not activity_probs[batch_idx, stage_idx, tooth_idx]:
                        f.write(f"Tooth {tooth_idx+31}: Inactive\n")
                        continue
                    
                    trans = transforms_sequence[batch_idx, stage_idx, tooth_idx, :3]
                    rot = transforms_sequence[batch_idx, stage_idx, tooth_idx, 3:]
                    
                    for param_idx in range(6):
                        if not param_activity_probs[batch_idx, stage_idx, tooth_idx, param_idx]:
                            if param_idx < 3:
                                trans[param_idx] = 0.0
                            else:
                                rot[param_idx - 3] = 0.0
                    
                    R = rotation_matrix_from_euler(rot, order='xyz')
                    T = torch.eye(4, device=R.device)
                    T[:3, :3] = R
                    T[:3, 3] = trans
                    
                    f.write(f"Tooth {tooth_idx+31}:\n")
                    np.savetxt(f, T.cpu().numpy(), fmt='%.6f')
                    f.write("\n")
            logger.info(f"Saved transformations: {output_path}")

def save_transformations_excel(transforms_sequence, activity_logits, type_logits, param_activity_logits, output_dir, jaw_ids, true_num_stages, logger):
    """
    Save transformations to Excel files for each jaw.
    
    Args:
        transforms_sequence (torch.Tensor): Predicted transformations of shape (batch_size, max_stages, num_teeth, 6)
        activity_logits (torch.Tensor): Activity logits of shape (batch_size, max_stages, num_teeth)
        type_logits (torch.Tensor): Type logits of shape (batch_size, max_stages, num_teeth, 4)
        param_activity_logits (torch.Tensor): Parameter activity logits of shape (batch_size, max_stages, num_teeth, 6)
        output_dir (str): Directory to save Excel files
        jaw_ids (list): List of jaw IDs
        true_num_stages (torch.Tensor): True number of stages for each batch
        logger (logging.Logger): Logger for debugging
    """
    os.makedirs(output_dir, exist_ok=True)
    batch_size, max_stages, num_teeth, _ = transforms_sequence.shape
    activity_probs = torch.sigmoid(activity_logits) > 0.5
    type_probs = torch.softmax(type_logits, dim=-1)
    type_predictions = torch.argmax(type_probs, dim=-1)
    param_activity_probs = torch.sigmoid(param_activity_logits) > 0.5
    
    type_map = {0: 'None', 1: 'Translation', 2: 'Rotation', 3: 'Both'}
    
    for batch_idx in range(batch_size):
        jaw_id = jaw_ids[batch_idx]
        num_stages = true_num_stages[batch_idx].item()
        
        data = []
        for stage_idx in range(num_stages):
            for tooth_idx in range(num_teeth):
                if not activity_probs[batch_idx, stage_idx, tooth_idx]:
                    trans = torch.zeros(3, device=transforms_sequence.device)
                    rot = torch.zeros(3, device=transforms_sequence.device)
                else:
                    trans = transforms_sequence[batch_idx, stage_idx, tooth_idx, :3]
                    rot = transforms_sequence[batch_idx, stage_idx, tooth_idx, 3:]
                    
                    for param_idx in range(6):
                        if not param_activity_probs[batch_idx, stage_idx, tooth_idx, param_idx]:
                            if param_idx < 3:
                                trans[param_idx] = 0.0
                            else:
                                rot[param_idx - 3] = 0.0
                
                is_active = activity_probs[batch_idx, stage_idx, tooth_idx].item()
                transform_type = type_map[type_predictions[batch_idx, stage_idx, tooth_idx].item()]
                
                data.append({
                    'Jaw_ID': jaw_id,
                    'Stage': stage_idx + 1,
                    'Tooth_ID': tooth_idx + 31,
                    'Left/Right (mm)': round(trans[0].item(), 2),
                    'Forward/Backward (mm)': round(trans[1].item(), 2),
                    'Extrude/Intrude (mm)': round(trans[2].item(), 2),
                    'Buccal/Lingual (degrees)': round(rot[0].item(), 2),
                    'Mesial/Distal (degrees)': round(rot[1].item(), 2),
                    'Rotation (degrees)': round(rot[2].item(), 2),
                    'Is_Active': is_active,
                    'Transform_Type': transform_type
                })
        
        df = pd.DataFrame(data)
        output_path = os.path.join(output_dir, f'{jaw_id}_transformations.xlsx')
        df.to_excel(output_path, index=False)
        logger.info(f"Saved Excel file: {output_path}")

def main(args):
    logger = setup_logging(args.log_file)
    logger.info("Inference with the following arguments:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    torch.cuda.empty_cache()
    logger.info(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    
    num_teeth = 14
    d_model = num_teeth * args.embed_dim
    if d_model % args.n_head != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by n_head ({args.n_head}).")
    
    dataset = JawTeethDataset(
        args.data_dir,
        max_stages=args.max_stages,
        split='test',
        train_ratio=args.train_ratio,
        inference=True,
        log_file=args.log_file
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Number of batches: {len(data_loader)}")
    
    dgcnn = DGCNN(in_channels=13, embed_dim=args.embed_dim, num_teeth=14, k=10).to(device)
    transformer = StageTransformer(
        d_model=d_model,
        max_stages=args.max_stages,
        n_head=args.n_head,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers
    ).to(device)
    model = OrthoDGCNNModel(
        dgcnn,
        transformer,
        max_stages=args.max_stages,
        num_teeth=14,
        embed_dim=args.embed_dim,
        teacher_forcing=False
    ).to(device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    logger.info(f"Loaded model from {args.model_path}")
    
    total_loss = 0
    total_trans_loss = 0
    total_rot_loss = 0
    total_padded_loss = 0
    total_sparsity_loss = 0
    total_activity_loss = 0
    total_type_loss = 0
    total_zero_trans_loss = 0
    total_zero_rot_loss = 0
    total_param_activity_loss = 0
    num_batches = 0
    
    all_stages_vertices = []
    all_faces_list = []
    all_true_num_stages = []
    all_jaw_ids = []
    
    with torch.no_grad():
        for batch_idx, (cordinates, targets, vertices_list, faces_list, true_num_stages, jaw_ids) in enumerate(data_loader):
            cordinates = cordinates.to(device)
            targets = targets.to(device)
            true_num_stages = true_num_stages.to(device)
            
            if torch.isnan(cordinates).any() or torch.isinf(cordinates).any():
                logger.warning(f"Batch {batch_idx+1}: cordinates contains nan/inf")
                continue
            if torch.isnan(targets).any() or torch.isinf(targets).any():
                logger.warning(f"Batch {batch_idx+1}: targets contains nan/inf")
                targets = torch.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
            
            with torch.amp.autocast('cuda'):
                transforms_sequence, activity_logits, type_logits, param_activity_logits = model(cordinates)
                
                logger.info(f"Batch {batch_idx+1}: Transforms min={transforms_sequence.min().item():.4f}, max={transforms_sequence.max().item():.4f}")
                
                if torch.isnan(transforms_sequence).any() or torch.isinf(transforms_sequence).any():
                    logger.warning(f"Batch {batch_idx+1}: transforms_sequence contains nan/inf")
                    continue
                if torch.isnan(activity_logits).any() or torch.isinf(activity_logits).any():
                    logger.warning(f"Batch {batch_idx+1}: activity_logits contains nan/inf")
                    continue
                if torch.isnan(type_logits).any() or torch.isinf(type_logits).any():
                    logger.warning(f"Batch {batch_idx+1}: type_logits contains nan/inf")
                    continue
                if torch.isnan(param_activity_logits).any() or torch.isinf(param_activity_logits).any():
                    logger.warning(f"Batch {batch_idx+1}: param_activity_logits contains nan/inf")
                    continue
                
                activity_labels = torch.any(targets != 0, dim=-1).float().to(device)
                param_activity_labels = (targets != 0).float().to(device)
                trans_only = torch.any(targets[:, :, :, :3] != 0, dim=-1) & ~torch.any(targets[:, :, :, 3:] != 0, dim=-1)
                rot_only = ~torch.any(targets[:, :, :, :3] != 0, dim=-1) & torch.any(targets[:, :, :, 3:] != 0, dim=-1)
                both = torch.any(targets[:, :, :, :3] != 0, dim=-1) & torch.any(targets[:, :, :, 3:] != 0, dim=-1)
                type_labels = torch.zeros_like(activity_labels, dtype=torch.long)
                type_labels[trans_only] = 1
                type_labels[rot_only] = 2
                type_labels[both] = 3
                type_labels = type_labels.to(device)
                
                loss, trans_loss, rot_loss, padded_loss, sparsity_loss, activity_loss, type_loss, zero_trans_loss, zero_rot_loss, param_activity_loss = compute_loss(
                    transforms_sequence, activity_logits, type_logits, param_activity_logits, targets, activity_labels,
                    type_labels, param_activity_labels, true_num_stages, args.max_stages, device, logger
                )
            
            total_loss += loss.item()
            total_trans_loss += trans_loss.item()
            total_rot_loss += rot_loss.item()
            total_padded_loss += padded_loss.item()
            total_sparsity_loss += sparsity_loss.item()
            total_activity_loss += activity_loss.item()
            total_type_loss += type_loss.item()
            total_zero_trans_loss += zero_trans_loss.item()
            total_zero_rot_loss += zero_rot_loss.item()
            total_param_activity_loss += param_activity_loss.item()
            num_batches += 1
            
            logger.info(f"Batch {batch_idx+1}: Total Loss = {loss.item():.6f}, "
                       f"Translation Loss = {trans_loss.item():.6f}, Rotation Loss = {rot_loss.item():.6f}, "
                       f"Zero Translation Loss = {zero_trans_loss.item():.6f}, Zero Rotation Loss = {zero_rot_loss.item():.6f}, "
                       f"Padded Loss = {padded_loss.item():.6f}, Sparsity Loss = {sparsity_loss.item():.6f}, "
                       f"Activity Loss = {activity_loss.item():.6f}, Type Loss = {type_loss.item():.6f}, "
                       f"Param Activity Loss = {param_activity_loss.item():.6f}")
            
            # stages_vertices = apply_transformations(vertices_list, transforms_sequence, activity_logits, param_activity_logits, logger)
            # all_stages_vertices.append(stages_vertices)
            # all_faces_list.append(faces_list)
            all_true_num_stages.append(true_num_stages.cpu())
            all_jaw_ids.extend(jaw_ids)
            
            save_transformations(transforms_sequence, activity_logits, param_activity_logits, args.output_dir, logger)
            save_transformations_excel(transforms_sequence, activity_logits, type_logits, param_activity_logits, args.output_dir, jaw_ids, true_num_stages, logger)
        
        # generate_stl_files(all_stages_vertices, all_faces_list, all_true_num_stages, args.output_dir, logger)
        
        if num_batches > 0:
            avg_total_loss = total_loss / num_batches
            avg_trans_loss = total_trans_loss / num_batches
            avg_rot_loss = total_rot_loss / num_batches
            avg_padded_loss = total_padded_loss / num_batches
            avg_sparsity_loss = total_sparsity_loss / num_batches
            avg_activity_loss = total_activity_loss / num_batches
            avg_type_loss = total_type_loss / num_batches
            avg_zero_trans_loss = total_zero_trans_loss / num_batches
            avg_zero_rot_loss = total_zero_rot_loss / num_batches
            avg_param_activity_loss = total_param_activity_loss / num_batches
            
            logger.info(f"Average Inference Results: Total Loss: {avg_total_loss:.4f}, "
                       f"Translation Loss: {avg_trans_loss:.4f}, Rotation Loss: {avg_rot_loss:.4f}, "
                       f"Zero Translation Loss: {avg_zero_trans_loss:.4f}, Zero Rotation Loss: {avg_zero_rot_loss:.4f}, "
                       f"Padded Loss: {avg_padded_loss:.4f}, Sparsity Loss: {avg_sparsity_loss:.4f}, "
                       f"Activity Loss: {avg_activity_loss:.4f}, Type Loss: {avg_type_loss:.4f}, "
                       f"Param Activity Loss: {avg_param_activity_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for Orthodontic Transformation Prediction")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the dataset')
    parser.add_argument('--model_path', type=str, default='/kaggle/working/Ortho_DGCNN_transformer/output/ortho_dgcnn.pth', help='Path to the trained model')
    parser.add_argument('--output_dir', type=str, default='inference_output', help='Directory to save inference outputs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for inference')
    parser.add_argument('--max_stages', type=int, default=25, help='Maximum number of treatment stages')
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension for DGCNN')
    parser.add_argument('--n_head', type=int, default=32, help='Number of attention heads in Transformer')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='Number of encoder layers in Transformer')
    parser.add_argument('--num_decoder_layers', type=int, default=6, help='Number of decoder layers in Transformer')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of data used for training')
    parser.add_argument('--log_file', type=str, default='inference_log.txt', help='Path to the log file')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)