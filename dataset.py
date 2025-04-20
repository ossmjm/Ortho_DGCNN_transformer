import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import time
import psutil
import tracemalloc

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
    def __init__(self, data_dir, max_stages=25, num_patches=128, patch_size=32, channels=13, split='train', train_ratio=0.8, inference=False, log_file='training_log.txt'):
        self.data_dir = data_dir
        self.max_stages = max_stages
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.channels = channels
        self.num_teeth = 14
        self.inference = inference
        
        self.logger = setup_logging(log_file)
        
        # Load num_stages.xlsx
        self.num_stages_df = pd.read_excel(os.path.join(data_dir, "num_stages.xlsx"))
        self.num_stages_df["Jaw_ID"] = self.num_stages_df["Jaw_ID"].astype(str).str.lstrip('0')
        self.num_stages_dict = dict(zip(self.num_stages_df["Jaw_ID"], self.num_stages_df["Num_Stages"]))
        self.logger.info(f"num_stages_dict: {self.num_stages_dict}")
        
        # Get cases
        self.cases = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()]
        self.logger.info(f"cases: {self.cases}")
        train_cases, test_cases = train_test_split(self.cases, train_size=train_ratio, random_state=42)
        self.cases = train_cases if split == 'train' else test_cases
        
        # Preload transformation files to reduce I/O
        self.logger.info("Preloading transformation files")
        transform_cache = {}
        for case in self.cases:
            transform_file = os.path.join(data_dir, case, "Transformations.xlsx")
            if os.path.exists(transform_file):
                try:
                    transform_cache[case] = pd.read_excel(transform_file, dtype={"Jaw_ID": str, "Tooth_ID": str})
                except Exception as e:
                    self.logger.warning(f"Failed to load {transform_file}: {e}")
        
        # Initialize lists
        self.json_files = []
        self.transformations = []
        self.activity_labels = []
        self.transform_types = []
        self.param_activity_labels = []
        original_cases = self.cases.copy()
        
        # Start memory tracking
        tracemalloc.start()
        
        self.logger.info(f"Processing {len(self.cases)} cases for split '{split}'")
        
        for case_idx, case in enumerate(original_cases):
            start_time = time.time()
            case_dir = os.path.join(data_dir, case)
            json_file = os.path.join(case_dir, "ori", "before_treatment.json")
            
            # Validate files
            if not os.path.exists(json_file):
                self.logger.warning(f"Skipping case {case}: Missing JSON file {json_file}")
                continue
            if case not in transform_cache:
                self.logger.warning(f"Skipping case {case}: No transformation data available")
                continue
            
            self.logger.debug(f"Processing case {case} ({case_idx+1}/{len(original_cases)})")
            
            # Load transformations from cache
            try:
                transformations = self._load_transformations(transform_cache[case], case)
            except Exception as e:
                self.logger.error(f"Error loading transformations for case {case}: {e}")
                continue
            
            # Compute labels
            activity = torch.any(transformations != 0, dim=-1).float()  # Shape: [max_stages, num_teeth]
            param_activity = (transformations != 0).float()  # Shape: [max_stages, num_teeth, 6]
            trans_only = torch.any(transformations[:, :, :3] != 0, dim=-1) & ~torch.any(transformations[:, :, 3:] != 0, dim=-1)
            rot_only = ~torch.any(transformations[:, :, :3] != 0, dim=-1) & torch.any(transformations[:, :, 3:] != 0, dim=-1)
            both = torch.any(transformations[:, :, :3] != 0, dim=-1) & torch.any(transformations[:, :, 3:] != 0, dim=-1)
            transform_type = torch.zeros_like(activity, dtype=torch.long)
            transform_type[trans_only] = 1
            transform_type[rot_only] = 2
            transform_type[both] = 3
            
            # Check for NaN/inf
            if torch.isnan(transformations).any() or torch.isinf(transformations).any():
                self.logger.warning(f"Transformations for Jaw_ID {case} contain NaN/inf. Replacing with zeros.")
                transformations = torch.nan_to_num(transformations, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Log sparsity
            zero_count = (activity == 0).sum().item()
            total_entries = activity.numel()
            zero_percentage = (zero_count / total_entries) * 100
            self.logger.info(f"Jaw_ID {case}: {zero_count}/{total_entries} tooth-stages are inactive ({zero_percentage:.2f}%)")
            
            # Append original data
            self.transformations.append(transformations)
            self.activity_labels.append(activity)
            self.transform_types.append(transform_type)
            self.param_activity_labels.append(param_activity)
            self.json_files.append(json_file)
            self.cases.append(case)
            
            # Data augmentation for training
            if not inference and split == 'train':
                active_teeth = torch.any(activity, dim=0).nonzero(as_tuple=True)[0]
                self.logger.debug(f"Case {case}: Found {len(active_teeth)} active teeth")
                
                for tooth_idx in active_teeth:
                    active_stages = activity[:, tooth_idx].nonzero(as_tuple=True)[0]
                    num_active_stages = len(active_stages)
                    if num_active_stages >= max_stages:
                        self.logger.debug(f"Case {case}, Tooth {tooth_idx}: Already has {num_active_stages} stages, skipping augmentation")
                        continue
                    
                    # Limit synthetic stages
                    num_synthetic = min(2, max_stages - num_active_stages)  # Reduced from 5 to 2
                    self.logger.debug(f"Case {case}, Tooth {tooth_idx}: Adding {num_synthetic} synthetic stages")
                    
                    for synth_idx in range(num_synthetic):
                        # Check total dataset size to prevent excessive growth
                        if len(self.transformations) >= len(original_cases) * 5:
                            self.logger.info("Reached maximum augmented dataset size, stopping augmentation")
                            break
                        
                        # Create synthetic transformations
                        synthetic_transforms = torch.zeros(1, self.num_teeth, 6)
                        num_active_params = np.random.randint(1, 4)
                        active_params = np.random.choice(6, num_active_params, replace=False)
                        for param_idx in active_params:
                            if param_idx < 3:  # Translation
                                synthetic_transforms[0, tooth_idx, param_idx] = np.random.uniform(-15, 15)
                            else:  # Rotation
                                synthetic_transforms[0, tooth_idx, param_idx] = np.random.uniform(-45, 45)
                        
                        # Compute activity labels
                        synthetic_activity = torch.any(synthetic_transforms != 0, dim=-1).float()  # Shape: [1, num_teeth]
                        synthetic_param_activity = (synthetic_transforms != 0).float()  # Shape: [1, num_teeth, 6]
                        
                        # Determine transformation type
                        synthetic_type = torch.zeros(1, self.num_teeth, dtype=torch.long)
                        trans_only = torch.any(synthetic_transforms[:, :, :3] != 0, dim=-1) & ~torch.any(synthetic_transforms[:, :, 3:] != 0, dim=-1)
                        rot_only = ~torch.any(synthetic_transforms[:, :, :3] != 0, dim=-1) & torch.any(synthetic_transforms[:, :, 3:] != 0, dim=-1)
                        both = torch.any(synthetic_transforms[:, :, :3] != 0, dim=-1) & torch.any(synthetic_transforms[:, :, 3:] != 0, dim=-1)
                        synthetic_type[trans_only] = 1
                        synthetic_type[rot_only] = 2
                        synthetic_type[both] = 3
                        
                        # Append augmented data
                        self.transformations.append(torch.cat([transformations, synthetic_transforms], dim=0)[:max_stages])
                        self.activity_labels.append(torch.cat([activity, synthetic_activity], dim=0)[:max_stages])
                        self.transform_types.append(torch.cat([transform_type, synthetic_type], dim=0)[:max_stages])
                        self.param_activity_labels.append(torch.cat([param_activity, synthetic_param_activity], dim=0)[:max_stages])
                        self.json_files.append(json_file)
                        self.cases.append(case)
                        
                        # Log memory usage
                        current, peak = tracemalloc.get_traced_memory()
                        process = psutil.Process()
                        mem_info = process.memory_info()
                        self.logger.debug(f"Case {case}, Tooth {tooth_idx}, Synthetic {synth_idx+1}: "
                                        f"Memory: Current={current/1e6:.2f}MB, Peak={peak/1e6:.2f}MB, "
                                        f"RSS={mem_info.rss/1e6:.2f}MB")
                
                # Check if augmentation was stopped early
                if len(self.transformations) >= len(original_cases) * 5:
                    break
            
            # Log completion of case
            elapsed = time.time() - start_time
            self.logger.info(f"Finished case {case} in {elapsed:.2f} seconds")
        
        # Stack tensors
        self.logger.info(f"Stacking {len(self.transformations)} transformation tensors")
        self.transformations = torch.stack(self.transformations)
        self.activity_labels = torch.stack(self.activity_labels)
        self.transform_types = torch.stack(self.transform_types)
        self.param_activity_labels = torch.stack(self.param_activity_labels)
        
        # Stop memory tracking
        tracemalloc.stop()
        
        self.logger.info(f"Dataset initialized with {len(self.cases)} cases (including {len(self.cases)-len(original_cases)} augmented)")

    def _load_transformations(self, transform_df, jaw_id):
        transformations = torch.zeros(self.max_stages, self.num_teeth, 6)
        FDI_TO_INDEX = {"31": 0, "32": 1, "33": 2, "34": 3, "35": 4, "36": 5, "37": 6,
                        "41": 7, "42": 8, "43": 9, "44": 10, "45": 11, "46": 12, "47": 13}
        
        jaw_data = transform_df[transform_df["Jaw_ID"] == jaw_id]
        if jaw_data.empty:
            self.logger.warning(f"No data found for Jaw_ID {jaw_id}")
            return transformations
        
        transform_columns = ["Left/Right (mm", "Forward/Backward (mm)", "Extrude/Intrude (mm)",
                            "Buccal/Lingual (degrees)", "Mesial/Distal (degrees)", "Rotation (degrees)"]
        for col in transform_columns:
            nan_count = jaw_data[col].isna().sum()
            if nan_count > 0:
                self.logger.info(f"Jaw_ID {jaw_id}, Column {col} has {nan_count} NaN values")
            inf_count = jaw_data[col].isin([float('inf'), -float('inf')]).sum()
            if inf_count > 0:
                self.logger.info(f"Jaw_ID {jaw_id}, Column {col} has {inf_count} inf values")
        
        for col in transform_columns:
            if jaw_data[col].isna().any() or jaw_data[col].isin([float('inf'), -float('inf')]).any():
                self.logger.warning(f"Column {col} for Jaw_ID {jaw_id} contains nan/inf. Replacing with 0.")
                jaw_data[col] = jaw_data[col].fillna(0).replace([float('inf'), -float('inf')], 0)
        
        jaw_data = jaw_data.copy()
        jaw_data["Stage"] = jaw_data["Stage"].astype("float64")
        
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
        
        jaw_data["Stage"] = jaw_data["Stage"].astype("Int64")
        
        if jaw_data["Stage"].isna().any():
            raise ValueError(f"After imputation, Stage column for Jaw_ID {jaw_id} still contains NaN values")
        
        invalid_stages = jaw_data[jaw_data["Stage"] > self.max_stages]["Stage"].unique()
        if invalid_stages.size > 0:
            self.logger.warning(f"Skipping transformations for Jaw_ID {jaw_id} with Stage values {invalid_stages} exceeding max_stages ({self.max_stages})")
        
        for stage in jaw_data["Stage"].unique():
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
        
        transformations = self.transformations[idx]
        activity = self.activity_labels[idx]
        transform_type = self.transform_types[idx]
        param_activity = self.param_activity_labels[idx]
        
        self.logger.debug(f"Jaw_ID: {jaw_id}, feats shape: {np.stack(feats_list).shape}, "
                         f"transformations shape: {transformations.shape}, activity shape: {activity.shape}, "
                         f"transform_type shape: {transform_type.shape}, param_activity shape: {param_activity.shape}, "
                         f"num_stages: {num_stages}")
        
        if self.inference:
            return (torch.tensor(np.stack(feats_list), dtype=torch.float32),
                    transformations,
                    vertices_list,
                    faces_list,
                    torch.tensor(num_stages, dtype=torch.int),
                    jaw_id)
        else:
            return (torch.tensor(np.stack(feats_list), dtype=torch.float32),
                    transformations,
                    torch.tensor(num_stages, dtype=torch.int),
                    activity,
                    transform_type,
                    param_activity)
    
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