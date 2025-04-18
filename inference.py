# inference.py
import argparse
import trimesh
import numpy as np
import pandas as pd
import torch
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

def apply_transformations(vertices_list, transforms_sequence, logger):
    stages_vertices = []
    batch_size = transforms_sequence.shape[0]
    
    for batch_idx in range(batch_size):
        jaw_stages = []
        current_vertices = [np.array(vertices_list[batch_idx][tooth_idx]) for tooth_idx in range(14)]
        
        for stage_idx in range(transforms_sequence.shape[1]):
            stage_vertices = []
            for tooth_idx in range(14):
                verts = current_vertices[tooth_idx].copy()
                transform = transforms_sequence[batch_idx][stage_idx, tooth_idx].cpu().numpy()
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

def save_transformations(transforms_sequence, output_dir, logger):
    os.makedirs(output_dir, exist_ok=True)
    batch_size = transforms_sequence.shape[0]
    for batch_idx in range(batch_size):
        for stage_idx in range(transforms_sequence.shape[1]):
            transform_matrix = transforms_sequence[batch_idx][stage_idx].cpu().numpy()
            output_path = f"{output_dir}/jaw_{batch_idx}_stage_{stage_idx+1}_transform.txt"
            np.savetxt(output_path, transform_matrix)
            logger.info(f"Saved transformation matrix to {output_path}")

def generate_stl_files(stages_vertices, faces_list, Fs_list, output_dir, logger):
    os.makedirs(output_dir, exist_ok=True)
    for batch_idx, jaw_stages in enumerate(stages_vertices):
        num_teeth = len(jaw_stages[0])
        batch_faces = faces_list[batch_idx]
        batch_Fs = Fs_list[batch_idx]
        for stage_idx, stage_vertices in enumerate(jaw_stages):
            all_vertices = []
            all_faces = []
            vertex_offset = 0
            for tooth_idx in range(num_teeth):
                tooth_vertices = stage_vertices[tooth_idx]
                num_faces = batch_Fs[tooth_idx].item()
                tooth_faces = batch_faces[tooth_idx][:num_faces]
                all_vertices.append(tooth_vertices)
                adjusted_faces = tooth_faces + vertex_offset
                all_faces.append(adjusted_faces)
                vertex_offset += len(tooth_vertices)
            
            all_vertices = np.concatenate(all_vertices, axis=0)
            all_faces = np.concatenate(all_faces, axis=0)
            
            stage_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
            output_path = f"{output_dir}/jaw_{batch_idx}_stage_{stage_idx+1}.stl"
            stage_mesh.export(output_path)
            logger.info(f"Saved STL file to {output_path}")

def save_transformations_excel(transforms_sequence, output_dir, jaw_ids=None, true_num_stages=None, logger=None):
    os.makedirs(output_dir, exist_ok=True)
    FDI_TO_INDEX = {"31": 0, "32": 1, "33": 2, "34": 3, "35": 4, "36": 5, "37": 6,
                    "41": 7, "42": 8, "43": 9, "44": 10, "45": 11, "46": 12, "47": 13}
    INDEX_TO_FDI = {v: k for k, v in FDI_TO_INDEX.items()}
    
    batch_size = transforms_sequence.shape[0]
    output_paths = []
    
    for batch_idx in range(batch_size):
        all_data = []
        jaw_id = jaw_ids[batch_idx] if jaw_ids else f"Jaw_{batch_idx:03d}"
        predicted_stages = 0
        
        for stage_idx in range(transforms_sequence.shape[1]):
            stage_transforms = transforms_sequence[batch_idx][stage_idx].cpu().numpy()
            threshold = np.max(np.abs(stage_transforms)) * 0.01  # Dynamic threshold (Enhancement 2)
            if np.all(np.abs(stage_transforms) < threshold):
                continue
            predicted_stages += 1
            for tooth_idx in range(14):
                transform = stage_transforms[tooth_idx]
                row = {
                    "Jaw_ID": jaw_id,
                    "Stage": stage_idx + 1,
                    "Tooth_ID": INDEX_TO_FDI[tooth_idx],
                    "Left/Right (mm)": transform[0],
                    "Forward/Backward (mm)": transform[1],
                    "Extrude/Intrude (mm)": transform[2],
                    "Buccal/Lingual (degrees)": transform[3],
                    "Mesial/Distal (degrees)": transform[4],
                    "Rotation (degrees)": transform[5]
                }
                all_data.append(row)
        
        # Log predicted vs. true stages (Enhancement 5)
        true_stages = true_num_stages[batch_idx].item() if true_num_stages is not None else "N/A"
        logger.info(f"Jaw {jaw_id}: Predicted stages = {predicted_stages}, True stages = {true_stages}")
        
        if all_data:
            df = pd.DataFrame(all_data)
            output_path = os.path.join(output_dir, f"Predicted_Transformations_{jaw_id}.xlsx")
            df.to_excel(output_path, index=False)
            logger.info(f"Saved transformations for {jaw_id} to {output_path}")
            output_paths.append(output_path)
    
    return output_paths

def inference(args):
    logger = setup_logging(args.log_file)
    logger.info("Starting inference with the following arguments:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
        logger.info(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    
    test_dataset = JawTeethDataset(
        args.data_dir, 
        max_stages=args.max_stages, 
        split='test', 
        train_ratio=args.train_ratio,
        inference=True,
        log_file=args.log_file
    )
    
    total_cases = len([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d)) and d.isdigit()])
    test_size = len(test_dataset)
    
    if test_size == 0:
        logger.error("Test dataset is empty. Check data_dir or train_ratio.")
        raise ValueError("Test dataset is empty. Check data_dir or train_ratio.")
    
    logger.info(f"Total cases in data_dir: {total_cases}")
    logger.info(f"Test set size (5% of total): {test_size}")
    
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    dgcnn = DGCNN(in_channels=13, embed_dim=args.embed_dim, num_teeth=14, k=10).to(device)
    transformer = StageTransformer(
        d_model=14 * args.embed_dim,
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
        embed_dim=args.embed_dim
    ).to(device)
    model_path = os.path.join(args.output_dir, "ortho_dgcnn.pth")
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    except RuntimeError as e:
        logger.error(f"Failed to load model state dict: {e}")
        logger.error(f"Ensure the checkpoint was trained with embed_dim={args.embed_dim} and in_channels=13.")
        raise
    model.eval()
    logger.info(f"Loaded model from {model_path}")
    logger.info(f"Model GPU memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    
    all_transforms = []
    all_jaw_ids = []
    all_true_stages = []
    
    with torch.no_grad():
        for batch_idx, (cordinates, _, vertices_list, faces_list, true_num_stages) in enumerate(dataloader):
            cordinates = cordinates.to(device)
            
            logger.info(f"Batch {batch_idx+1}: Input cordinates shape: {cordinates.shape}")
            logger.info(f"Batch {batch_idx+1}: True num_stages: {true_num_stages.tolist()}")
            logger.info(f"Batch {batch_idx+1}: GPU memory before inference: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
            
            with torch.amp.autocast('cuda'):
                transforms_sequence = model(cordinates)
            
            logger.info(f"Batch {batch_idx+1}: transforms_sequence shape: {transforms_sequence.shape}")
            logger.info(f"Batch {batch_idx+1}: GPU memory after inference: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
            
            save_transformations(transforms_sequence, args.output_dir, logger)
            
            # stages_vertices = apply_transformations(vertices_list, transforms_sequence, logger)
            # generate_stl_files(stages_vertices, faces_list, true_num_stages, args.output_dir, logger)
            
            all_transforms.append(transforms_sequence.cpu())
            all_jaw_ids.extend([f"Jaw_{batch_idx * args.batch_size + i:03d}" 
                               for i in range(transforms_sequence.shape[0])])
            all_true_stages.extend(true_num_stages.tolist())
            
            del cordinates, transforms_sequence
            torch.cuda.empty_cache()
            logger.info(f"Batch {batch_idx+1}: GPU memory after cleanup: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
            
            logger.info(f"Inference complete for {len(vertices_list)} jaws in batch {batch_idx+1}")
    
    all_transforms = torch.cat(all_transforms, dim=0)
    excel_paths = save_transformations_excel(all_transforms, args.output_dir, jaw_ids=all_jaw_ids, true_num_stages=all_true_stages, logger=logger)
    for path in excel_paths:
        logger.info(f"Excel output saved at: {path}")
    logger.info("Inference completed successfully.")
    logger.info(f"Final GPU memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with OrthoDGCNNModel")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--max_stages', type=int, default=25)
    parser.add_argument('--train_ratio', type=float, default=0.95)
    parser.add_argument('--output_dir', type=str, default="output")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_head', type=int, default=16)
    parser.add_argument('--num_encoder_layers', type=int, default=6)
    parser.add_argument('--num_decoder_layers', type=int, default=6)
    parser.add_argument('--log_file', type=str, default="inference_log.txt")
    
    args = parser.parse_args()
    
    log_dir = os.path.dirname(args.log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    logger = setup_logging(args.log_file)
    logger.info("Initializing inference script...")
    
    inference(args)