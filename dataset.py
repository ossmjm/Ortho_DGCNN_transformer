# JawTeethDataset.ipynb
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import trimesh
from sklearn.model_selection import train_test_split

class JawTeethDataset(Dataset):
    def __init__(self, data_dir, max_stages=20, num_patches=256, patch_size=64, channels=13, split='train', train_ratio=0.8, inference=False):
        self.data_dir = data_dir
        self.max_stages = max_stages
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.channels = channels
        self.num_teeth = 14
        self.inference = inference  # New parameter to control output
        
        self.num_stages_df = pd.read_excel(os.path.join(data_dir, "num_stages.xlsx"))
        self.num_stages_dict = dict(zip(self.num_stages_df["Jaw_ID"], self.num_stages_df["Num_Stages"]))
        
        self.cases = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()]
        train_cases, test_cases = train_test_split(self.cases, train_size=train_ratio, random_state=42)
        self.cases = train_cases if split == 'train' else test_cases
        
        self.stl_files = []
        self.json_files = []
        self.transformations = []
        for case in self.cases:
            case_dir = os.path.join(data_dir, case)
            stl_file = os.path.join(case_dir, "ori", "before_treatment.stl")
            json_file = os.path.join(case_dir, "ori", "before_treatment.json")
            transform_file = os.path.join(case_dir, "Transformations.xlsx")
            self.stl_files.append(stl_file)
            self.json_files.append(json_file)
            self.transformations.append(self._load_transformations(transform_file, case))
        self.transformations = torch.stack(self.transformations)

    def _load_transformations(self, transform_file, jaw_id):
        transform_df = pd.read_excel(transform_file)
        transformations = torch.zeros(self.max_stages, self.num_teeth, 6)
        FDI_TO_INDEX = {"31": 0, "32": 1, "33": 2, "34": 3, "35": 4, "36": 5, "37": 6,
                        "41": 7, "42": 8, "43": 9, "44": 10, "45": 11, "46": 12, "47": 13}
        
        jaw_data = transform_df[transform_df["Jaw_ID"] == jaw_id]
        for stage in jaw_data["Stage"].unique():
            stage_data = jaw_data[jaw_data["Stage"] == stage]
            for _, row in stage_data.iterrows():
                tooth_idx = FDI_TO_INDEX[str(int(row["Tooth_ID"]))]
                transformations[stage - 1, tooth_idx] = torch.tensor([
                    row["Left/Right (mm"], row["Forward/Backward (mm)"], row["Extrude/Intrude (mm)"],
                    row["Buccal/Lingual (degrees)"], row["Mesial/Distal (degrees)"], row["Rotation (degrees)"]
                ])
        return transformations

    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        stl_file = self.stl_files[idx]
        json_file = self.json_files[idx]
        jaw_id = self.cases[idx]
        mesh = trimesh.load(stl_file)
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        FDI_TO_INDEX = {"31": 0, "32": 1, "33": 2, "34": 3, "35": 4, "36": 5, "37": 6,
                        "41": 7, "42": 8, "43": 9, "44": 10, "45": 11, "46": 12, "47": 13}
        
        faces_list, cordinates_list, vertices_list = [None] * 14, [None] * 14, [None] * 14
        teeth_data = data["teeth"]
        
        # Check for missing teeth and handle them
        for fdi in FDI_TO_INDEX.keys():
            tooth_idx = FDI_TO_INDEX[fdi]
            if fdi not in teeth_data:
                print(f"Warning: Tooth {fdi} missing in JSON file {json_file}")
                # Create placeholder data for the missing tooth
                total_points = self.num_patches * self.patch_size
                vertices = np.zeros((total_points, 3))  # Zero-filled vertices
                faces = np.array([[0, 0, 0]])  # Single dummy face
                cordinates = vertices
            else:
                tooth_data = teeth_data[fdi]
                vertices = np.array(tooth_data["v"])
                faces = np.array(tooth_data["f"])
                _, _, _, cordinates = self.preprocess_tooth_points(vertices, faces)
            
            faces_list[tooth_idx], cordinates_list[tooth_idx], vertices_list[tooth_idx] = faces, cordinates, vertices
        
        num_stages = self.num_stages_dict.get(jaw_id, self.max_stages)
        
        # Return different elements based on inference flag
        if self.inference:
            return (torch.tensor(np.stack(cordinates_list), dtype=torch.float32),  # cordinates
                    self.transformations[idx],  # targets
                    vertices_list,  # Needed for inference
                    faces_list,  # Needed for inference
                    torch.tensor(num_stages, dtype=torch.int))  # true_num_stages
        else:
            return (torch.tensor(np.stack(cordinates_list), dtype=torch.float32),  # cordinates
                    self.transformations[idx],  # targets
                    torch.tensor(num_stages, dtype=torch.int))  # true_num_stages

    def preprocess_tooth_points(self, vertices, faces):
        vertices = np.array(vertices)
        faces = np.array(faces)
        
        # Sample points if there are too many vertices
        total_points = self.num_patches * self.patch_size
        if len(vertices) > total_points:
            sampled_indices = np.random.choice(len(vertices), total_points, replace=False)
            vertices = vertices[sampled_indices]
            # Adjust face indices to match sampled vertices
            vertex_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sampled_indices)}
            faces = np.array([[vertex_mapping.get(idx, 0) for idx in face] for face in faces if all(idx in vertex_mapping for idx in face)])
        
        # Ensure we have enough points
        if len(vertices) < total_points:
            vertices = np.pad(vertices, ((0, total_points - len(vertices)), (0, 0)), mode='edge')
        
        # Organize points into patches
        feats = np.zeros((self.num_patches, self.channels, self.patch_size))
        centers = np.zeros((self.num_patches, self.patch_size, 3))
        for i in range(self.num_patches):
            start = i * self.patch_size
            end = min((i + 1) * self.patch_size, len(vertices))
            patch_points = vertices[start:end]
            if len(patch_points) < self.patch_size:
                patch_points = np.pad(patch_points, ((0, self.patch_size - len(patch_points)), (0, 0)), mode='edge')
            feats[i, :3, :] = patch_points.T  # First 3 channels are x, y, z
            centers[i] = patch_points
        
        # Number of faces
        Fs = len(faces) if len(faces) > 0 else 1  # Ensure Fs is at least 1
        
        return feats, centers, Fs, vertices