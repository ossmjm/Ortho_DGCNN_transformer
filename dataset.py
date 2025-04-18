import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import logging
import json

def setup_logging(log_file):
    logger = logging.getLogger('DatasetLogger')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

class JawTeethDataset(Dataset):
    def __init__(self, data_dir, max_stages=25, split='train', train_ratio=0.95, inference=False, log_file='training_log.txt'):
        self.data_dir = data_dir
        self.max_stages = max_stages
        self.split = split
        self.train_ratio = train_ratio
        self.inference = inference
        self.num_teeth = 14
        self.num_patches = 128
        self.patch_size = 32
        self.channels = 13
        
        # Initialize logger
        self.logger = setup_logging(log_file)
        
        # Load cases
        self.cases = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()]
        self.cases.sort()
        
        if split == 'train':
            self.cases = self.cases[:int(len(self.cases) * train_ratio)]
        else:
            self.cases = self.cases[int(len(self.cases) * train_ratio):]
        
        # Compute number of stages per case
        self.num_stages_dict = {}
        for case in self.cases:
            df = pd.read_excel(os.path.join(data_dir, case, "Transformations.xlsx"))
            num_stages = len(df['Stage'].dropna().unique())
            self.num_stages_dict[case] = num_stages
            if inference:
                zero_entries = (df.iloc[:, 3:9] == 0).all(axis=1).sum()
                total_entries = len(df)
                self.logger.info(f"Transformations for Jaw_ID {case}: {zero_entries}/{total_entries} entries are all zeros ({100 * zero_entries / total_entries:.2f}%)")
    
    def __len__(self):
        return len(self.cases)
    
    def _load_transformations(self, transform_file, jaw_id):
        # Load Excel file with flexible typing
        transform_df = pd.read_excel(transform_file, dtype={"Jaw_ID": str, "Tooth_ID": str, "Stage": "float64"})
        transformations = torch.zeros(self.max_stages, self.num_teeth, 6, dtype=torch.float32)
        FDI_TO_INDEX = {
            "31": 0, "32": 1, "33": 2, "34": 3, "35": 4, "36": 5, "37": 6,
            "41": 7, "42": 8, "43": 9, "44": 10, "45": 11, "46": 12, "47": 13
        }
        
        jaw_data = transform_df[transform_df["Jaw_ID"] == jaw_id]
        if jaw_data.empty:
            self.logger.warning(f"No data found for Jaw_ID {jaw_id} in {transform_file}")
            return transformations
        
        # Check for NaN/inf in transformation columns
        transform_columns = [
            "Left/Right (mm)", "Forward/Backward (mm)", "Extrude/Intrude (mm)",
            "Buccal/Lingual (degrees)", "Mesial/Distal (degrees)", "Rotation (degrees)"
        ]
        for col in transform_columns:
            nan_count = jaw_data[col].isna().sum()
            if nan_count > 0:
                self.logger.info(f"Jaw_ID {jaw_id}, Column {col} has {nan_count} NaN values")
            inf_count = jaw_data[col].isin([float('inf'), -float('inf')]).sum()
            if inf_count > 0:
                self.logger.info(f"Jaw_ID {jaw_id}, Column {col} has {inf_count} inf values")
        
        # Replace NaN/inf with 0
        jaw_data = jaw_data.copy()
        for col in transform_columns:
            if jaw_data[col].isna().any() or jaw_data[col].isin([float('inf'), -float('inf')]).any():
                self.logger.warning(f"Column {col} for Jaw_ID {jaw_id} contains NaN/inf. Replacing with 0.")
                jaw_data[col] = jaw_data[col].fillna(0).replace([float('inf'), -float('inf')], 0)
        
        # Impute NaN Stage values
        for idx in jaw_data.index:
            if idx == jaw_data.index[0]:
                if pd.isna(jaw_data.at[idx, "Stage"]):
                    imputed_stage = 1
                    self.logger.info(f"Imputing NaN Stage at index {idx} for Jaw_ID {jaw_id}, Tooth_ID {jaw_data.at[idx, 'Tooth_ID']}: First row, defaulting to Stage {imputed_stage}")
                    jaw_data.at[idx, "Stage"] = imputed_stage
            else:
                if pd.isna(jaw_data.at[idx, "Stage"]):
                    prev_stage = jaw_data.at[idx - 1, "Stage"]
                    tooth_id = jaw_data.at[idx, "Tooth_ID"]
                    if tooth_id != "31":
                        imputed_stage = prev_stage
                        self.logger.info(f"Imputing NaN Stage at index {idx} for Jaw_ID {jaw_id}, Tooth_ID {tooth_id}: Using previous Stage {imputed_stage}")
                    else:
                        imputed_stage = prev_stage + 1
                        self.logger.info(f"Imputing NaN Stage at index {idx} for Jaw_ID {jaw_id}, Tooth_ID {tooth_id}: Tooth_ID is 31, using previous Stage {prev_stage} + 1 = {imputed_stage}")
                    jaw_data.at[idx, "Stage"] = imputed_stage
        
        # Convert Stage to nullable integer
        jaw_data["Stage"] = jaw_data["Stage"].astype("Int64")
        
        if jaw_data["Stage"].isna().any():
            raise ValueError(f"After imputation, Stage column for Jaw_ID {jaw_id} in {transform_file} still contains NaN values")
        
        # Validate Stage values
        invalid_stages = jaw_data[jaw_data["Stage"] > self.max_stages]["Stage"].unique()
        if invalid_stages.size > 0:
            self.logger.warning(f"Skipping transformations for Jaw_ID {jaw_id} with Stage values {invalid_stages} exceeding max_stages ({self.max_stages})")
        
        # Load transformations
        for stage in jaw_data["Stage"].dropna().unique():
            stage = int(stage)
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
                
                # Clean transformation values
                def clean_and_convert(value):
                    try:
                        value_str = str(value).replace('o', '0').replace('O', '0')
                        return float(value_str)
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Error converting value '{value}' to float: {e}. Setting to 0.")
                        return 0.0
                
                transform_values = torch.tensor([
                    clean_and_convert(row["Left/Right (mm)"]),
                    clean_and_convert(row["Forward/Backward (mm)"]),
                    clean_and_convert(row["Extrude/Intrude (mm)"]),
                    clean_and_convert(row["Buccal/Lingual (degrees)"]),
                    clean_and_convert(row["Mesial/Distal (degrees)"]),
                    clean_and_convert(row["Rotation (degrees)"])
                ], dtype=torch.float32)
                
                if torch.isnan(transform_values).any() or torch.isinf(transform_values).any():
                    self.logger.info(f"Jaw_ID {jaw_id}, Stage {stage}, Tooth_ID {tooth_id}: transform_values contains NaN/inf: {transform_values.tolist()}")
                
                transformations[stage - 1, tooth_idx] = transform_values
        
        return transformations
    
    def __getitem__(self, idx):
        case = self.cases[idx]
        case_dir = os.path.join(self.data_dir, case)
        json_file = os.path.join(case_dir, "ori", "before_treatment.json")
        
        # Load JSON file
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading JSON file {json_file}: {e}")
            raise
        
        FDI_TO_INDEX = {
            "31": 0, "32": 1, "33": 2, "34": 3, "35": 4, "36": 5, "37": 6,
            "41": 7, "42": 8, "43": 9, "44": 10, "45": 11, "46": 12, "47": 13
        }
        
        vertices_list = []
        faces_list = []
        feats_list = []
        
        teeth_data = data.get("teeth", {})
        for fdi in FDI_TO_INDEX.keys():
            tooth_idx = FDI_TO_INDEX[fdi]
            total_points = self.num_patches * self.patch_size  # 128 * 32 = 4096
            if fdi not in teeth_data:
                self.logger.warning(f"Tooth {fdi} missing in JSON file {json_file}")
                vertices = np.zeros((total_points, 3))
                faces = np.array([[0, 0, 0]])
                feats = np.zeros((self.num_patches, self.channels, self.patch_size))
            else:
                tooth_data = teeth_data[fdi]
                vertices = np.array(tooth_data.get("v", []))
                faces = np.array(tooth_data.get("f", []))
                if len(vertices) == 0 or len(faces) == 0:
                    self.logger.warning(f"Tooth {fdi} in {json_file} has empty vertices or faces")
                    vertices = np.zeros((total_points, 3))
                    faces = np.array([[0, 0, 0]])
                    feats = np.zeros((self.num_patches, self.channels, self.patch_size))
                else:
                    # Standardize vertices to exactly total_points
                    if len(vertices) > total_points:
                        indices = np.random.choice(len(vertices), total_points, replace=False)
                        vertices = vertices[indices]
                        vertex_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}
                        faces = np.array([[vertex_mapping.get(idx, 0) for idx in face]
                                        for face in faces if all(idx in vertex_mapping for idx in face)])
                    elif len(vertices) < total_points:
                        pad_size = total_points - len(vertices)
                        vertices = np.pad(vertices, ((0, pad_size), (0, 0)), mode='edge')
                    else:
                        vertices = vertices  # Already correct size
                    
                    # Create feature tensor with only xyz coordinates
                    feats = np.zeros((self.num_patches, self.channels, self.patch_size))
                    for i in range(self.num_patches):
                        start = i * self.patch_size
                        end = (i + 1) * self.patch_size
                        patch_points = vertices[start:end]
                        if len(patch_points) < self.patch_size:
                            pad_size = self.patch_size - len(patch_points)
                            patch_points = np.pad(patch_points, ((0, pad_size), (0, 0)), mode='edge')
                        feats[i, :3, :] = patch_points.T  # x, y, z
                        # Other channels remain zero
            
            vertices_list.append(vertices)
            faces_list.append(faces)
            feats_list.append(feats)
        
        # Stack feats into cordinates
        cordinates = np.stack(feats_list)
        cordinates = torch.tensor(cordinates, dtype=torch.float32)
        
        true_num_stages = self.num_stages_dict[case]
        if not self.inference:
            transform_file = os.path.join(case_dir, "Transformations.xlsx")
            transformations = self._load_transformations(transform_file, case)
            zero_entries = (transformations == 0).all(dim=-1).sum().item()
            total_entries = transformations.shape[0] * transformations.shape[1]
            self.logger.info(f"Transformations for Jaw_ID {case}: {zero_entries}/{total_entries} entries are all zeros ({100 * zero_entries / total_entries:.2f}%)")
        else:
            transformations = torch.zeros(self.max_stages, self.num_teeth, 6, dtype=torch.float32)
        
        faces_list = [torch.tensor(faces, dtype=torch.long) for faces in faces_list]
        
        return cordinates, transformations, vertices_list, faces_list, true_num_stages