import argparse
import trimesh
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import logging
from dataset import JawTeethDataset
from models.DGCNN import DGCNN
from models.StageTransformer import StageTransformer
from models.OrthoDGCNN import OrthoDGCNNModel

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

def log_cosh_loss(pred, target, amplify_threshold=0.5, amplify_factor=10.0):
    diff = pred - target
    loss = torch.log(torch.cosh(diff + 1e-12))
    small_error_mask = (torch.abs(diff) < amplify_threshold).float()
    amplified_loss = loss * (1 + amplify_factor * small_error_mask)
    return torch.mean(amplified_loss)

def zero_prediction_loss(pred, target, threshold=0.1):
    zero_mask = (target == 0).float()
    non_zero_pred = torch.abs(pred) * zero_mask
    return torch.mean(torch.relu(non_zero_pred - threshold) ** 2)

def compute_loss(transforms_sequence, activity_logits, type_logits, param_activity_logits, targets, activity_labels, type_labels, param_activity_labels, true_num_stages, max_stages, device, logger):
    loss_trans = log_cosh_loss(transforms_sequence[:, :, :, :3], targets[:, :, :, :3])
    loss_rot = log_cosh_loss(transforms_sequence[:, :, :, 3:], targets[:, :, :, 3:])
    
    zero_trans_loss = zero_prediction_loss(transforms_sequence[:, :, :, :3], targets[:, :, :, :3])
    zero_rot_loss = zero_prediction_loss(transforms_sequence[:, :, :, 3:], targets[:, :, :, 3:])
    
    bce = nn.BCEWithLogitsLoss(reduction='none')
    loss_activity = bce(activity_logits, activity_labels)
    loss_param_activity = bce(param_activity_logits, param_activity_labels)
    
    ce = nn.CrossEntropyLoss(reduction='none')
    type_logits_flat = type_logits.view(-1, 4)
    type_labels_flat = type_labels.view(-1)
    loss_type = ce(type_logits_flat, type_labels_flat)
    
    batch_size = transforms_sequence.size(0)
    stage_weights = torch.zeros(batch_size, max_stages, device=device)
    for b in range(batch_size):
        stage_weights[b, :true_num_stages[b]] = 1.0
    
    active_mask = activity_labels
    loss_trans = (loss_trans * active_mask * stage_weights.unsqueeze(-1)).sum() / (active_mask * stage_weights.unsqueeze(-1)).sum().clamp(min=1)
    loss_rot = (loss_rot * active_mask * stage_weights.unsqueeze(-1)).sum() / (active_mask * stage_weights.unsqueeze(-1)).sum().clamp(min=1)
    zero_trans_loss = (zero_trans_loss * active_mask * stage_weights.unsqueeze(-1)).sum() / (active_mask * stage_weights.unsqueeze(-1)).sum().clamp(min=1)
    zero_rot_loss = (zero_rot_loss * active_mask * stage_weights.unsqueeze(-1)).sum() / (active_mask * stage_weights.unsqueeze(-1)).sum().clamp(min=1)
    loss_activity = (loss_activity * stage_weights.unsqueeze(-1)).mean()
    loss_param_activity = (loss_param_activity * stage_weights.unsqueeze(-1).unsqueeze(-1)).mean()
    loss_type = (loss_type.view(batch_size, max_stages, -1) * stage_weights.unsqueeze(-1)).mean()
    
    padded_loss = 0.0
    for b in range(batch_size):
        true_stages = true_num_stages[b].item()
        if true_stages < max_stages:
            padded_loss += torch.mean(transforms_sequence[b, true_stages:, :, :]**2)
    padded_loss = padded_loss / batch_size if batch_size > 0 else 0.0
    
    sparsity_loss = torch.mean((transforms_sequence * (1 - param_activity_labels))**2)
    
    activity_preds = (torch.sigmoid(activity_logits) > 0.5).float()
    activity_accuracy = (activity_preds == activity_labels).float().mean()
    type_preds = torch.argmax(type_logits, dim=-1)
    type_accuracy = (type_preds == type_labels).float().mean()
    param_activity_preds = (torch.sigmoid(param_activity_logits) > 0.5).float()
    param_activity_accuracy = (param_activity_preds == param_activity_labels).float().mean()
    logger.debug(f"Batch Activity prediction accuracy: {activity_accuracy:.4f}")
    logger.debug(f"Batch Type prediction accuracy: {type_accuracy:.4f}")
    logger.debug(f"Batch Param Activity prediction accuracy: {param_activity_accuracy:.4f}")
    
    alpha, beta, gamma, delta, epsilon, zeta, eta, theta = 10.0, 50.0, 0.1, 30.0, 10.0, 20.0, 5.0, 20.0
    total_loss = (alpha * loss_trans + beta * loss_rot + gamma * padded_loss + 
                  delta * sparsity_loss + epsilon * loss_activity + zeta * (zero_trans_loss + zero_rot_loss) + 
                  eta * loss_type + theta * loss_param_activity)
    
    return total_loss, loss_trans, loss_rot, padded_loss, sparsity_loss, loss_activity, loss_type, zero_trans_loss, zero_rot_loss, loss_param_activity

def apply_transformations(vertices_list, transforms_sequence, activity_logits, param_activity_logits, logger):
    activity_probs = torch.sigmoid(activity_logits).cpu().numpy()
    param_activity_probs = torch.sigmoid(param_activity_logits).cpu().numpy()
    transforms_sequence = transforms_sequence.cpu().numpy()
    inactive_mask = activity_probs < 0.5
    param_inactive_mask = param_activity_probs < 0.5
    transforms_sequence[inactive_mask] = 0
    transforms_sequence[param_inactive_mask] = 0
    
    stages_vertices = []
    batch_size = transforms_sequence.shape[0]
    num_teeth = 14
    FDI_TO_INDEX = {"31": 0, "32": 1, "33": 2, "34": 3, "35": 4, "36": 5, "37": 6,
                    "41": 7, "42": 8, "43": 9, "44": 10, "45": 11, "46": 12, "47": 13}
    INDEX_TO_FDI = {v: k for k, v in FDI_TO_INDEX.items()}
    
    for batch_idx in range(batch_size):
        jaw_stages = []
        current_vertices = [None] * num_teeth
        for tooth_idx in range(num_teeth):
            try:
                vertices = vertices_list[batch_idx][tooth_idx]
                if vertices is not None and len(vertices) > 0:
                    current_vertices[tooth_idx] = np.array(vertices)
                else:
                    current_vertices[tooth_idx] = np.zeros((4096, 3))
                    logger.warning(f"Batch {batch_idx}, Tooth {INDEX_TO_FDI[tooth_idx]}: Missing vertices, using zero placeholder")
            except (IndexError, TypeError):
                current_vertices[tooth_idx] = np.zeros((4096, 3))
                logger.warning(f"Batch {batch_idx}, Tooth {INDEX_TO_FDI[tooth_idx]}: Missing vertices, using zero placeholder")
        
        for stage_idx in range(transforms_sequence.shape[1]):
            stage_vertices = []
            stage_transforms = np.round(transforms_sequence[batch_idx][stage_idx], 2)
            for tooth_idx in range(num_teeth):
                if current_vertices[tooth_idx].sum() == 0:
                    stage_vertices.append(current_vertices[tooth_idx])
                    continue
                verts = current_vertices[tooth_idx].copy()
                transform = stage_transforms[tooth_idx]
                centroid = np.mean(verts, axis=0)
                translation = transform[:3]
                verts += translation
                rotations = np.radians(transform[3:])
                if rotations[0] != 0:
                    rot_x = trimesh.transformations.rotation_matrix(rotations[0], [1, 0, 0], point=centroid)
                    verts = trimesh.transformations.transform_points(verts, rot_x)
                if rotations[1] != 0:
                    rot_y = trimesh.transformations.rotation_matrix(rotations[1], [0, 1, 0], point=centroid)
                    verts = trimesh.transformations.transform_points(verts, rot_y)
                if rotations[2] != 0:
                    rot_z = trimesh.transformations.rotation_matrix(rotations[2], [0, 0, 1], point=centroid)
                    verts = trimesh.transformations.transform_points(verts, rot_z)
                stage_vertices.append(verts)
                current_vertices[tooth_idx] = verts
            jaw_stages.append(stage_vertices)
        stages_vertices.append(jaw_stages)
    return stages_vertices

def save_transformations(transforms_sequence, activity_logits, param_activity_logits, output_dir, logger):
    os.makedirs(output_dir, exist_ok=True)
    activity_probs = torch.sigmoid(activity_logits).cpu().numpy()
    param_activity_probs = torch.sigmoid(param_activity_logits).cpu().numpy()
    transforms_sequence = transforms_sequence.cpu().numpy()
    inactive_mask = activity_probs < 0.5
    param_inactive_mask = param_activity_probs < 0.5
    transforms_sequence[inactive_mask] = 0
    transforms_sequence[param_inactive_mask] = 0
    
    batch_size = transforms_sequence.shape[0]
    for batch_idx in range(batch_size):
        for stage_idx in range(transforms_sequence.shape[1]):
            transform_matrix = np.round(transforms_sequence[batch_idx][stage_idx], 2)
            output_path = f"{output_dir}/jaw_{batch_idx}_stage_{stage_idx+1}_transform.txt"
            np.savetxt(output_path, transform_matrix)
            logger.info(f"Saved transformation matrix to {output_path}")

def generate_stl_files(stages_vertices, faces_list, true_num_stages, output_dir, logger):
    os.makedirs(output_dir, exist_ok=True)
    num_teeth = 14
    FDI_TO_INDEX = {"31": 0, "32": 1, "33": 2, "34": 3, "35": 4, "36": 5, "37": 6,
                    "41": 7, "42": 8, "43": 9, "44": 10, "45": 11, "46": 12, "47": 13}
    INDEX_TO_FDI = {v: k for k, v in FDI_TO_INDEX.items()}
    
    for batch_idx, jaw_stages in enumerate(stages_vertices):
        batch_faces = faces_list[batch_idx]
        true_stages = true_num_stages[batch_idx].item()
        for stage_idx in range(true_stages):
            all_vertices = []
            all_faces = []
            vertex_offset = 0
            for tooth_idx in range(num_teeth):
                tooth_vertices = jaw_stages[stage_idx][tooth_idx]
                try:
                    tooth_faces = batch_faces[tooth_idx]
                    if tooth_faces is None or len(tooth_faces) == 0 or tooth_vertices.sum() == 0:
                        logger.warning(f"Batch {batch_idx}, Tooth {INDEX_TO_FDI[tooth_idx]}, Stage {stage_idx+1}: Skipping STL due to missing faces or vertices")
                        continue
                except (IndexError, TypeError):
                    logger.warning(f"Batch {batch_idx}, Tooth {INDEX_TO_FDI[tooth_idx]}, Stage {stage_idx+1}: Skipping STL due to missing faces")
                    continue
                all_vertices.append(tooth_vertices)
                adjusted_faces = tooth_faces + vertex_offset
                all_faces.append(adjusted_faces)
                vertex_offset += len(tooth_vertices)
            
            if not all_vertices or not all_faces:
                logger.warning(f"Batch {batch_idx}, Stage {stage_idx+1}: No valid vertices or faces, skipping STL")
                continue
            
            all_vertices = np.concatenate(all_vertices, axis=0)
            all_faces = np.concatenate(all_faces, axis=0)
            
            stage_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
            output_path = f"{output_dir}/jaw_{batch_idx}_stage_{stage_idx+1}.stl"
            stage_mesh.export(output_path)
            logger.info(f"Saved STL file to {output_path}")

def save_transformations_excel(transforms_sequence, activity_logits, type_logits, param_activity_logits, output_dir, jaw_ids=None, true_num_stages=None, logger=None):
    os.makedirs(output_dir, exist_ok=True)
    FDI_TO_INDEX = {"31": 0, "32": 1, "33": 2, "34": 3, "35": 4, "36": 5, "37": 6,
                    "41": 7, "42": 8, "43": 9, "44": 10, "45": 11, "46": 12, "47": 13}
    INDEX_TO_FDI = {v: k for k, v in FDI_TO_INDEX.items()}
    TYPE_TO_STR = {0: "None", 1: "Translation", 2: "Rotation", 3: "Both"}
    
    activity_probs = torch.sigmoid(activity_logits).cpu().numpy()
    param_activity_probs = torch.sigmoid(param_activity_logits).cpu().numpy()
    transforms_sequence = transforms_sequence.cpu().numpy()
    type_preds = torch.argmax(type_logits, dim=-1).cpu().numpy()
    inactive_mask = activity_probs < 0.5
    param_inactive_mask = param_activity_probs < 0.5
    transforms_sequence[inactive_mask] = 0
    transforms_sequence[param_inactive_mask] = 0
    
    batch_size = transforms_sequence.shape[0]
    output_paths = []
    
    for batch_idx in range(batch_size):
        all_data = []
        jaw_id = jaw_ids[batch_idx] if jaw_ids else f"Jaw_{batch_idx:03d}"
        true_stages = true_num_stages[batch_idx].item() if true_num_stages is not None else transforms_sequence.shape[1]
        
        for stage_idx in range(true_stages):
            stage_transforms = np.round(transforms_sequence[batch_idx][stage_idx], 2)
            stage_activity = activity_probs[batch_idx][stage_idx]
            stage_types = type_preds[batch_idx][stage_idx]
            stage_param_activity = param_activity_probs[batch_idx][stage_idx]
            
            for tooth_idx in range(transforms_sequence.shape[2]):
                tooth_id = INDEX_TO_FDI[tooth_idx]
                transform = stage_transforms[tooth_idx]
                is_active = stage_activity[tooth_idx] >= 0.5
                transform_type = TYPE_TO_STR[stage_types[tooth_idx]]
                param_active = stage_param_activity[tooth_idx] >= 0.5
                
                data_row = {
                    "Jaw_ID": jaw_id,
                    "Stage": stage_idx + 1,
                    "Tooth_ID": tooth_id,
                    "Left/Right (mm)": transform[0] if param_active[0] else 0,
                    "Forward/Backward (mm)": transform[1] if param_active[1] else 0,
                    "Extrude/Intrude (mm)": transform[2] if param_active[2] else 0,
                    "Buccal/Lingual (degrees)": transform[3] if param_active[3] else 0,
                    "Mesial/Distal (degrees)": transform[4] if param_active[4] else 0,
                    "Rotation (degrees)": transform[5] if param_active[5] else 0,
                    "Is_Active": is_active,
                    "Transform_Type": transform_type
                }
                all_data.append(data_row)
        
        df = pd.DataFrame(all_data)
        output_path = os.path.join(output_dir, f"{jaw_id}_transformations.xlsx")
        df.to_excel(output_path, index=False)
        output_paths.append(output_path)
        logger.info(f"Saved transformations to {output_path}")
    
    return output_paths

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
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
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
                
                # Compute activity and type labels for loss
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
    parser = argparse.ArgumentParser(description="Run inference with OrthoDGCNN model.")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing the dataset")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model")
    parser.add_argument('--output_dir', type=str, default="inference_output", help="Directory to save outputs")
    parser.add_argument('--max_stages', type=int, default=25, help="Maximum number of stages")
    parser.add_argument('--train_ratio', type=float, default=0.8, help="Train/test split ratio")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for inference")
    parser.add_argument('--embed_dim', type=int, default=256, help="Embedding dimension")
    parser.add_argument('--n_head', type=int, default=32, help="Number of attention heads")
    parser.add_argument('--num_encoder_layers', type=int, default=6, help="Number of encoder layers")
    parser.add_argument('--num_decoder_layers', type=int, default=6, help="Number of decoder layers")
    parser.add_argument('--log_file', type=str, default="inference_log.txt", help="Path to log file")
    
    args = parser.parse_args()
    
    log_dir = os.path.dirname(args.log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    main(args)