import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

# Set up logging
def setup_logging(log_file):
    logger = logging.getLogger('DatasetLogger')
    logger.setLevel(logging.INFO)
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Create formatters and add them to handlers
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class JawTeethDataset(Dataset):
    def __init__(self, data_dir, max_stages=25, num_patches=128, patch_size=32, channels=13, split='train', train_ratio=0.8, inference=False, log_file='training_log.txt'):
        self.data_dir = data_dir
        self.max_stages = max_stages
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.channels = channels
        self.num_teeth = 14
        self.inference = inference
        
        # Initialize logger
        self.logger = setup_logging(log_file)
        
        # Load num_stages.xlsx and normalize Jaw_ID
        self.num_stages_df = pd.read_excel(os.path.join(data_dir, "num_stages.xlsx"))
        self.num_stages_df["Jaw_ID"] = self.num_stages_df["Jaw_ID"].astype(str).str.lstrip('0')
        self.num_stages_dict = dict(zip(self.num_stages_df["Jaw_ID"], self.num_stages_df["Num_Stages"]))
        self.logger.info(f"num_stages_dict: {self.num_stages_dict}")
        
        self.cases = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()]
        self.logger.info(f"cases: {self.cases}")
        train_cases, test_cases = train_test_split(self.cases, train_size=train_ratio, random_state=42)
        self.cases = train_cases if split == 'train' else test_cases
        
        self.json_files = []
        self.transformations = []
        for case in self.cases:
            case_dir = os.path.join(data_dir, case)
            json_file = os.path.join(case_dir, "ori", "before_treatment.json")
            transform_file = os.path.join(case_dir, "Transformations.xlsx")
            self.json_files.append(json_file)
            transformations = self._load_transformations(transform_file, case)
            # Debug: Check for nan/inf in transformations
            if torch.isnan(transformations).any() or torch.isinf(transformations).any():
                self.logger.warning(f"transformations for Jaw_ID {case} contains nan/inf")
            # Debug: Check for unexpected zeros (possible missing data)
            zero_mask = (transformations == 0).all(dim=-1)  # Check if all 6 transformation values are 0
            zero_count = zero_mask.sum().item()
            total_entries = zero_mask.numel()
            zero_percentage = (zero_count / total_entries) * 100
            self.logger.info(f"transformations for Jaw_ID {case}: {zero_count}/{total_entries} entries are all zeros ({zero_percentage:.2f}%)")
            self.transformations.append(transformations)
        self.transformations = torch.stack(self.transformations)

    def _load_transformations(self, transform_file, jaw_id):
        # Load the Excel file without forcing Stage to int, allowing NaN values
        transform_df = pd.read_excel(transform_file, dtype={"Jaw_ID": str, "Tooth_ID": str})
        transformations = torch.zeros(self.max_stages, self.num_teeth, 6)
        FDI_TO_INDEX = {"31": 0, "32": 1, "33": 2, "34": 3, "35": 4, "36": 5, "37": 6,
                        "41": 7, "42": 8, "43": 9, "44": 10, "45": 11, "46": 12, "47": 13}
        
        jaw_data = transform_df[transform_df["Jaw_ID"] == jaw_id]
        if jaw_data.empty:
            self.logger.warning(f"No data found for Jaw_ID {jaw_id} in {transform_file}")
            return transformations
        
        # Debug: Check for missing values in raw Excel data
        transform_columns = ["Left/Right (mm", "Forward/Backward (mm)", "Extrude/Intrude (mm)",
                            "Buccal/Lingual (degrees)", "Mesial/Distal (degrees)", "Rotation (degrees)"]
        for col in transform_columns:
            # Check for NaN
            nan_count = jaw_data[col].isna().sum()
            if nan_count > 0:
                self.logger.info(f"Jaw_ID {jaw_id}, Column {col} has {nan_count} NaN values")
            # Check for inf
            inf_count = jaw_data[col].isin([float('inf'), -float('inf')]).sum()
            if inf_count > 0:
                self.logger.info(f"Jaw_ID {jaw_id}, Column {col} has {inf_count} inf values")
        
        # Replace nan/inf with 0
        for col in transform_columns:
            if jaw_data[col].isna().any() or jaw_data[col].isin([float('inf'), -float('inf')]).any():
                self.logger.warning(f"Column {col} for Jaw_ID {jaw_id} contains nan/inf. Replacing with 0.")
                jaw_data[col] = jaw_data[col].fillna(0).replace([float('inf'), -float('inf')], 0)
        
        # Impute NaN Stage values by looking at the previous row
        jaw_data = jaw_data.copy()  # Avoid SettingWithCopyWarning
        jaw_data["Stage"] = jaw_data["Stage"].astype("float64")  # Ensure Stage is float to handle NaN
        
        # Process rows sequentially to impute NaN Stage values
        for idx in jaw_data.index:
            if idx == jaw_data.index[0]:
                # Handle the first row
                if pd.isna(jaw_data.at[idx, "Stage"]):
                    imputed_stage = 1  # Default to Stage 1 for the first row
                    self.logger.info(f"Imputing NaN Stage at index {idx} for Jaw_ID {jaw_id}, Tooth_ID {jaw_data.at[idx, 'Tooth_ID']}: First row, defaulting to Stage {imputed_stage}")
                    jaw_data.at[idx, "Stage"] = imputed_stage
            else:
                # Handle subsequent rows
                if pd.isna(jaw_data.at[idx, "Stage"]):
                    prev_stage = jaw_data.at[idx - 1, "Stage"]  # Stage of the previous row
                    tooth_id = jaw_data.at[idx, "Tooth_ID"]
                    if tooth_id != "31":
                        # If Tooth_ID is not 31, use the previous row's Stage
                        imputed_stage = prev_stage
                        self.logger.info(f"Imputing NaN Stage at index {idx} for Jaw_ID {jaw_id}, Tooth_ID {tooth_id}: Using previous Stage {imputed_stage}")
                    else:
                        # If Tooth_ID is 31, use the previous row's Stage + 1
                        imputed_stage = prev_stage + 1
                        self.logger.info(f"Imputing NaN Stage at index {idx} for Jaw_ID {jaw_id}, Tooth_ID {tooth_id}: Tooth_ID is 31, using previous Stage {prev_stage} + 1 = {imputed_stage}")
                    jaw_data.at[idx, "Stage"] = imputed_stage
        
        # Convert Stage to nullable integer type Int64
        jaw_data["Stage"] = jaw_data["Stage"].astype("Int64")
        
        # Validate that all Stage values are now integers
        if jaw_data["Stage"].isna().any():
            raise ValueError(f"After imputation, Stage column for Jaw_ID {jaw_id} in {transform_file} still contains NaN values")
        
        # Validate Stage values are within bounds (1 to max_stages)
        invalid_stages = jaw_data[jaw_data["Stage"] > self.max_stages]["Stage"].unique()
        if invalid_stages.size > 0:
            self.logger.warning(f"Skipping transformations for Jaw_ID {jaw_id} with Stage values {invalid_stages} exceeding max_stages ({self.max_stages})")
        
        for stage in jaw_data["Stage"].unique():
            stage = int(stage)  # Ensure stage is an integer
            if stage > self.max_stages:
                continue
                
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
                        self.logger.info(f"Skipping Tooth_ID {tooth_id} as it is not in FDI_TO_INDEX.")
                        continue
                    tooth_idx = FDI_TO_INDEX[tooth_id]
                except Exception as e:
                    self.logger.error(f"Error processing Tooth_ID {tooth_id_raw}: {e}")
                    raise
                
                # Helper function to clean and convert transformation values
                def clean_and_convert(value):
                    try:
                        value_str = str(value).replace('o', '0').replace('O', '0')
                        return float(value_str)
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Error converting value '{value}' to float: {e}. Setting to 0.")
                        return 0.0
                
                transform_values = torch.tensor([
                    clean_and_convert(row["Left/Right (mm"]), 
                    clean_and_convert(row["Forward/Backward (mm)"]), 
                    clean_and_convert(row["Extrude/Intrude (mm)"]),
                    clean_and_convert(row["Buccal/Lingual (degrees)"]), 
                    clean_and_convert(row["Mesial/Distal (degrees)"]), 
                    clean_and_convert(row["Rotation (degrees)"])
                ], dtype=torch.float32)
                
                if torch.isnan(transform_values).any() or torch.isinf(transform_values).any():
                    self.logger.info(f"Jaw_ID {jaw_id}, Stage {stage}, Tooth_ID {tooth_id}: transform_values contains nan/inf: {transform_values.tolist()}")
                
                transformations[stage - 1, tooth_idx] = transform_values
        return transformations

    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        json_file = self.json_files[idx]
        jaw_id = self.cases[idx]
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        FDI_TO_INDEX = {"31": 0, "32": 1, "33": 2, "34": 3, "35": 4, "36": 5, "37": 6,
                        "41": 7, "42": 8, "43": 9, "44": 10, "45": 11, "46": 12, "47": 13}
        
        faces_list, feats_list, vertices_list = [None] * 14, [None] * 14, [None] * 14
        teeth_data = data["teeth"]
        
        for fdi in FDI_TO_INDEX.keys():
            tooth_idx = FDI_TO_INDEX[fdi]
            if fdi not in teeth_data:
                self.logger.warning(f"Tooth {fdi} missing in JSON file {json_file}")
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
        
        jaw_id_normalized = str(jaw_id).lstrip('0')
        num_stages = self.num_stages_dict.get(jaw_id_normalized, 1)
        num_stages = min(num_stages, self.max_stages)
        
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