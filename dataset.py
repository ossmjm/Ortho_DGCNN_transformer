# dataset.py
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import trimesh
from sklearn.model_selection import train_test_split

class JawTeethDataset(Dataset):
    def __init__(self, data_dir, max_stages=20, num_patches=128, patch_size=32, channels=13, split='train', train_ratio=0.8, inference=False):
        self.data_dir = data_dir
        self.max_stages = max_stages
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.channels = channels
        self.num_teeth = 14
        self.inference = inference
        
        # Load num_stages.xlsx and normalize Jaw_ID
        self.num_stages_df = pd.read_excel(os.path.join(data_dir, "num_stages.xlsx"))
        self.num_stages_df["Jaw_ID"] = self.num_stages_df["Jaw_ID"].astype(str).str.lstrip('0')
        self.num_stages_dict = dict(zip(self.num_stages_df["Jaw_ID"], self.num_stages_df["Num_Stages"]))
        print(f"num_stages_dict: {self.num_stages_dict}")
        
        self.cases = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()]
        print(f"cases: {self.cases}")
        train_cases, test_cases = train_test_split(self.cases, train_size=train_ratio, random_state=42)
        self.cases = train_cases if split == 'train' else test_cases
        
        self.stl_files = []
        self.json_files = []
        self.transformations = []
        self.num_stages_list = []
        for case in self.cases:
            case_dir = os.path.join(data_dir, case)
            stl_file = os.path.join(case_dir, "ori", "before_treatment.stl")
            json_file = os.path.join(case_dir, "ori", "before_treatment.json")
            transform_file = os.path.join(case_dir, "Transformations.xlsx")
            self.stl_files.append(stl_file)
            self.json_files.append(json_file)
            transformations = self._load_transformations(transform_file, case)
            self.transformations.append(transformations)
            num_stages = self._compute_num_stages(transformations)
            self.num_stages_list.append(num_stages)
        self.transformations = torch.stack(self.transformations)

    def _compute_num_stages(self, transformations):
        for stage in range(self.max_stages):
            if not torch.any(transformations[stage, :, :] != 0):
                return stage if stage > 0 else 1
        return self.max_stages

    def _load_transformations(self, transform_file, jaw_id):
        transform_df = pd.read_excel(transform_file, dtype={"Jaw_ID": str, "Tooth_ID": str})
        transformations = torch.zeros(self.max_stages, self.num_teeth, 6)
        FDI_TO_INDEX = {"31": 0, "32": 1, "33": 2, "34": 3, "35": 4, "36": 5, "37": 6,
                        "41": 7, "42": 8, "43": 9, "44": 10, "45": 11, "46": 12, "47": 13}
        
        jaw_id = str(jaw_id).lstrip('0')
        jaw_data = transform_df[transform_df["Jaw_ID"] == jaw_id]
        if jaw_data.empty:
            print(f"Warning: No data found for Jaw_ID {jaw_id} in {transform_file}")
            return transformations
        
        for stage in jaw_data["Stage"].unique():
            stage_data = jaw_data[jaw_data["Stage"] == stage]
            if stage_data.empty:
                continue
            
            for _, row in stage_data.iterrows():
                tooth_id_raw = row["Tooth_ID"]
                try:
                    tooth_id_clean = str(tooth_id_raw).strip().replace(',', '.')
                    tooth_id = tooth_id_clean.split('.')[0]
                    tooth_id = ''.join(filter(str.isdigit, tooth_id))
                    if tooth_id not in FDI_TO_INDEX:
                        print(f"Skipping Tooth_ID {tooth_id} as it is not in FDI_TO_INDEX.")
                        continue
                    tooth_idx = FDI_TO_INDEX[tooth_id]
                except Exception as e:
                    print(f"Error processing Tooth_ID {tooth_id_raw}: {e}")
                    raise
                
                transformations[stage - 1, tooth_idx] = torch.tensor([
                    float(row["Left/Right (mm"]), 
                    float(row["Forward/Backward (mm)"]), 
                    float(row["Extrude/Intrude (mm)"]),
                    float(row["Buccal/Lingual (degrees)"]), 
                    float(row["Mesial/Distal (degrees)"]), 
                    float(row["Rotation (degrees)"])
                ], dtype=torch.float32)
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
        
        faces_list, feats_list, vertices_list = [None] * 14, [None] * 14, [None] * 14
        teeth_data = data["teeth"]
        
        for fdi in FDI_TO_INDEX.keys():
            tooth_idx = FDI_TO_INDEX[fdi]
            if fdi not in teeth_data:
                print(f"Warning: Tooth {fdi} missing in JSON file {json_file}")
                total_points = self.num_patches * self.patch_size
                vertices = np.zeros((total_points, 3))
                faces = np.array([[0, 0, 0]])
                feats = np.zeros((self.num_patches, self.channels, self.patch_size))
            else:
                tooth_data = teeth_data[fdi]
                vertices = np.array(tooth_data["v"])
                faces = np.array(tooth_data["f"])
                feats, _, _, vertices = self.preprocess_tooth_points(vertices, faces)
            
            faces_list[tooth_idx], feats_list[tooth_idx], vertices_list[tooth_idx] = faces, feats, vertices
        
        num_stages = self.num_stages_list[idx]
        jaw_id_normalized = str(jaw_id).lstrip('0')
        dict_num_stages = self.num_stages_dict.get(jaw_id_normalized, None)
        print(f"actual num_stages :{dict_num_stages}")
        if dict_num_stages is not None:
            num_stages = min(dict_num_stages, self.max_stages)
        
        if self.inference:
            return (torch.tensor(np.stack(feats_list), dtype=torch.float32),
                    self.transformations[idx],
                    vertices_list,
                    faces_list,
                    torch.tensor(num_stages, dtype=torch.int))
        else:
            return (torch.tensor(np.stack(feats_list), dtype=torch.float32),
                    self.transformations[idx],
                    torch.tensor(num_stages, dtype=torch.int))

    def preprocess_tooth_points(self, vertices, faces):
        vertices = np.array(vertices)
        faces = np.array(faces)
        
        total_points = self.num_patches * self.patch_size
        if len(vertices) > total_points:
            sampled_indices = np.random.choice(len(vertices), total_points, replace=False)
            vertices = vertices[sampled_indices]
            vertex_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sampled_indices)}
            faces = np.array([[vertex_mapping.get(idx, 0) for idx in face] for face in faces if all(idx in vertex_mapping for idx in face)])
        
        if len(vertices) < total_points:
            vertices = np.pad(vertices, ((0, total_points - len(vertices)), (0, 0)), mode='edge')
        
        feats = np.zeros((self.num_patches, self.channels, self.patch_size))
        centers = np.zeros((self.num_patches, self.patch_size, 3))
        for i in range(self.num_patches):
            start = i * self.patch_size
            end = min((i + 1) * self.patch_size, len(vertices))
            patch_points = vertices[start:end]
            if len(patch_points) < self.patch_size:
                patch_points = np.pad(patch_points, ((0, self.patch_size - len(patch_points)), (0, 0)), mode='edge')
            feats[i, :3, :] = patch_points.T
            centers[i] = patch_points
        
        Fs = len(faces) if len(faces) > 0 else 1
        
        return feats, centers, Fs, vertices