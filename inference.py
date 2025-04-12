# inference.py
import argparse
import trimesh
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import os
from dataset import JawTeethDataset
from models.DGCNN import DGCNN
from models.StageTransformer import StageTransformer
from models.OrthoDGCNN import OrthoDGCNNModel

def apply_transformations(vertices_list, transforms_sequence, num_stages_pred):
    stages_vertices = []
    batch_size = transforms_sequence.shape[0]
    
    for batch_idx in range(batch_size):
        jaw_stages = []
        num_stages = num_stages_pred[batch_idx].item()  # Use predicted number of stages
        # Initialize current vertices as the original vertices for this batch
        current_vertices = [np.array(vertices_list[batch_idx][tooth_idx]) for tooth_idx in range(14)]
        
        for stage_idx in range(num_stages):
            stage_vertices = []
            for tooth_idx in range(14):
                verts = current_vertices[tooth_idx].copy()  # Work on a copy of the current vertices
                transform = transforms_sequence[batch_idx][stage_idx, tooth_idx].cpu().numpy()
                centroid = np.mean(verts, axis=0)
                translation = transform[:3]
                verts += translation
                rotations = np.radians(transform[3:])
                if rotations[0] != 0:  # Rx (around X-axis)
                    rot_x = trimesh.transformations.rotation_matrix(rotations[0], [1, 0, 0], point=centroid)
                    verts = trimesh.transformations.transform_points(verts, rot_x)
                if rotations[1] != 0:  # Ry (around Y-axis)
                    rot_y = trimesh.transformations.rotation_matrix(rotations[1], [0, 1, 0], point=centroid)
                    verts = trimesh.transformations.transform_points(verts, rot_y)
                if rotations[2] != 0:  # Rz (around Z-axis)
                    rot_z = trimesh.transformations.rotation_matrix(rotations[2], [0, 0, 1], point=centroid)
                    verts = trimesh.transformations.transform_points(verts, rot_z)
                stage_vertices.append(verts)
                # Update current_vertices for the next stage
                current_vertices[tooth_idx] = verts
            jaw_stages.append(stage_vertices)
        stages_vertices.append(jaw_stages)
    return stages_vertices

def save_transformations(transforms_sequence, output_dir, num_stages_pred):
    os.makedirs(output_dir, exist_ok=True)
    batch_size = transforms_sequence.shape[0]
    for batch_idx in range(batch_size):
        num_stages = num_stages_pred[batch_idx].item()
        for stage_idx in range(num_stages):
            transform_matrix = transforms_sequence[batch_idx][stage_idx].cpu().numpy()
            np.savetxt(f"{output_dir}/jaw_{batch_idx}_stage_{stage_idx+1}_transform.txt", transform_matrix)

def generate_stl_files(stages_vertices, faces_list, Fs_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for batch_idx, jaw_stages in enumerate(stages_vertices):
        num_teeth = len(jaw_stages[0])  # Should be 14
        # Get faces and Fs for this batch
        batch_faces = faces_list[batch_idx]  # Shape: (num_teeth, max_num_faces, 3)
        batch_Fs = Fs_list[batch_idx]  # Shape: (num_teeth,)
        for stage_idx, stage_vertices in enumerate(jaw_stages):
            # Combine vertices and faces for all teeth in this stage
            all_vertices = []
            all_faces = []
            vertex_offset = 0
            for tooth_idx in range(num_teeth):
                tooth_vertices = stage_vertices[tooth_idx]
                num_faces = batch_Fs[tooth_idx].item()  # Number of valid faces for this tooth
                tooth_faces = batch_faces[tooth_idx][:num_faces]  # Take only the valid faces
                all_vertices.append(tooth_vertices)
                # Adjust face indices for the concatenated vertex list
                adjusted_faces = tooth_faces + vertex_offset
                all_faces.append(adjusted_faces)
                vertex_offset += len(tooth_vertices)
            
            # Concatenate vertices and faces
            all_vertices = np.concatenate(all_vertices, axis=0)
            all_faces = np.concatenate(all_faces, axis=0)
            
            # Create and export the mesh
            stage_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
            stage_mesh.export(f"{output_dir}/jaw_{batch_idx}_stage_{stage_idx+1}.stl")

def save_transformations_excel(transforms_sequence, output_dir, num_stages_pred, jaw_ids=None):
    os.makedirs(output_dir, exist_ok=True)
    FDI_TO_INDEX = {"31": 0, "32": 1, "33": 2, "34": 3, "35": 4, "36": 5, "37": 6,
                    "41": 7, "42": 8, "43": 9, "44": 10, "45": 11, "46": 12, "47": 13}
    INDEX_TO_FDI = {v: k for k, v in FDI_TO_INDEX.items()}
    
    batch_size = transforms_sequence.shape[0]
    all_data = []
    
    for batch_idx in range(batch_size):
        jaw_id = jaw_ids[batch_idx] if jaw_ids else f"Jaw_{batch_idx:03d}"
        num_stages = num_stages_pred[batch_idx].item()
        for stage_idx in range(num_stages):
            for tooth_idx in range(14):
                transform = transforms_sequence[batch_idx][stage_idx, tooth_idx].cpu().numpy()
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
    
    df = pd.DataFrame(all_data)
    output_path = os.path.join(output_dir, "Predicted_Transformations.xlsx")
    df.to_excel(output_path, index=False)
    print(f"Saved transformations to {output_path}")

def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_dataset = JawTeethDataset(args.data_dir, max_stages=args.max_stages, split='test', train_ratio=args.train_ratio)
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    dgcnn = DGCNN(in_channels=3, embed_dim=512, num_teeth=14, k=20).to(device)
    transformer = StageTransformer(d_model=14 * 512, max_stages=args.max_stages).to(device)
    model = OrthoDGCNNModel(dgcnn, transformer, max_stages=args.max_stages).to(device)
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "ortho_dgcnn.pth")))
    model.eval()
    
    all_transforms = []
    all_jaw_ids = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            faces, feats, centers, Fs, cordinates, _, vertices_list, all_vertices, _ = batch
            faces, feats, centers, Fs, cordinates, all_vertices = [
                x.to(device) for x in [faces, feats, centers, Fs, cordinates, all_vertices]
            ]
            
            transforms_sequence, num_stages_pred = model(
                cordinates, feats, centers, Fs, faces, all_vertices
            )
            
            print(f"Batch {batch_idx+1}: transforms_sequence shape: {transforms_sequence.shape}, "
                  f"num_stages: {num_stages_pred.tolist()}")
            
            stages_vertices = apply_transformations(vertices_list, transforms_sequence, num_stages_pred)
            save_transformations(transforms_sequence, args.output_dir, num_stages_pred)
            generate_stl_files(stages_vertices, faces, Fs, args.output_dir)
            
            all_transforms.append(transforms_sequence)
            all_jaw_ids.extend([f"Jaw_{batch_idx * args.batch_size + i:03d}" 
                               for i in range(transforms_sequence.shape[0])])
            
            print(f"Inference complete for {len(vertices_list)} jaws in batch {batch_idx+1}, "
                  f"stages: {num_stages_pred.tolist()}")
    
    all_transforms = torch.cat(all_transforms, dim=0)
    save_transformations_excel(all_transforms, args.output_dir, num_stages_pred, jaw_ids=all_jaw_ids)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with OrthoDGCNNModel")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to data directory")
    parser.add_argument('--max_stages', type=int, default=20, help="Maximum number of stages")
    parser.add_argument('--train_ratio', type=float, default=0.8, help="Train split ratio")
    parser.add_argument('--output_dir', type=str, default="output", 
                        help="Directory to load OrthoDGCNNModel from and save outputs")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for inference")
    args = parser.parse_args()
    inference(args)