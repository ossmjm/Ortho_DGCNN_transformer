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
from models.DGCNN import DGCNN
from models.MViT import MViTv2
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

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with OrthoDGCNN Model")
    parser.add_argument('--data-dir', type=str, default='./data', help='Path to dataset directory')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for inference')
    parser.add_argument('--max-stages', type=int, default=25, help='Maximum number of stages')
    parser.add_argument('--log-file', type=str, default='inference_log.txt', help='Log file path')
    parser.add_argument('--model-path', type=str, default='best_model.pth', help='Path to trained model')
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory for STL files')
    parser.add_argument('--embed-dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--depths', type=str, default='[1, 2, 11, 2]', help='Number of blocks per stage')
    parser.add_argument('--num-heads', type=str, default='[4, 4, 8, 8]', help='Number of attention heads per stage')
    parser.add_argument('--mlp-ratio', type=float, default=4.0, help='MLP expansion ratio')
    parser.add_argument('--drop-path-rate', type=float, default=0.2, help='Drop path rate')
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
    
    num_teeth = 14
    FDI_numbers = [31, 32, 33, 34, 35, 36, 37, 41, 42, 43, 44, 45, 46, 47]
    depths = ast.literal_eval(args.depths)
    num_heads = ast.literal_eval(args.num_heads)
    
    dataset = JawTeethDataset(
        args.data_dir, 
        max_stages=args.max_stages, 
        split='test', 
        train_ratio=0.8, 
        inference=True,
        log_file=args.log_file,
    )
    logger.info(f"Dataset size: {len(dataset)}")
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    logger.info(f"Number of batches: {len(data_loader)}")
    
    dgcnn = DGCNN(in_channels=13, embed_dim=args.embed_dim, num_teeth=14, k=10).to(device)
    mvit = MViTv2(
        embed_dim=args.embed_dim,
        num_teeth=14,
        max_stages=args.max_stages,
        depths=depths,
        num_heads=num_heads,
        mlp_ratio=args.mlp_ratio,
        drop_path_rate=args.drop_path_rate,
        teacher_forcing=False
    ).to(device)
    model = OrthoDGCNNModel(
        dgcnn, 
        mvit, 
        max_stages=args.max_stages,
        num_teeth=14,
        embed_dim=args.embed_dim,
        teacher_forcing=False,
        depths=depths,
        num_heads=num_heads,
        mlp_ratio=args.mlp_ratio,
        drop_path_rate=args.drop_path_rate
    ).to(device)
    
    logger.info(f"Loading model from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    os.makedirs(args.output_dir, exist_ok=True)
    transform_data = []
    
    with torch.no_grad():
        for batch_idx, (cordinates, _, vertices_list, faces_list, num_stages, jaw_ids) in enumerate(data_loader):
            cordinates = cordinates.to(device)
            batch_size = cordinates.size(0)
            
            if torch.isnan(cordinates).any() or torch.isinf(cordinates).any():
                logger.warning(f"Batch {batch_idx+1}: cordinates contains nan/inf")
                continue
            
            with torch.amp.autocast('cuda'):
                transforms_sequence, _, _, _ = model(cordinates)  # [batch_size, max_stages, num_teeth, 6]
            
            if torch.isnan(transforms_sequence).any() or torch.isinf(transforms_sequence).any():
                logger.warning(f"Batch {batch_idx+1}: transforms_sequence contains nan/inf")
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
                        # vertices = vertices_list[b][tooth_idx]
                        # faces = faces_list[b][tooth_idx]
                        
                        # if len(vertices) == 0 or len(faces) == 0:
                        #     logger.warning(f"Jaw {jaw_id}, Stage {stage+1}, Tooth {fdi}: Empty vertices or faces")
                        #     continue
                        
                        # vertices_transformed = apply_transformations(vertices, translation, rotation)
                        # mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=faces, process=False)
                        # output_path = os.path.join(jaw_output_dir, f'stage_{stage+1}_tooth_{fdi}.stl')
                        # try:
                        #     mesh.export(output_path)
                        #     logger.info(f"Saved STL for Jaw {jaw_id}, Stage {stage+1}, Tooth {fdi} at {output_path}")
                        # except Exception as e:
                        #     logger.error(f"Error saving STL for Jaw {jaw_id}, Stage {stage+1}, Tooth {fdi}: {e}")
                        
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
    
    df = pd.DataFrame(transform_data)
    excel_path = os.path.join(args.output_dir, 'predicted_transformations.xlsx')
    try:
        df.to_excel(excel_path, index=False)
        logger.info(f"Saved transformation data to {excel_path}")
    except Exception as e:
        logger.error(f"Error saving Excel file: {e}")
    
    logger.info("Inference completed")

if __name__ == "__main__":
    args = parse_args()
    infer_model(args)