import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
import logging
import re
import open3d as o3d  # Added for curvature-aware sampling

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

def clean_numeric(value):
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).strip().lower()
    value = re.sub(r'[oO]', '0', value)
    value = re.sub(r'[^0-9.\-]', '', value)
    try:
        return float(value) if value else 0.0
    except ValueError:
        return 0.0

# Added function for curvature-aware sampling
def curvature_aware_sampling(points, n_points, logger):
    if len(points) <= n_points:
        return points
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # Estimate normals for curvature computation
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        pcd.estimate_covariances()
        curvatures = []
        for cov in pcd.covariances:
            eigvals = np.linalg.eigvals(cov)
            curvature = eigvals.min() / (eigvals.sum() + 1e-6)  # Approximate curvature
            curvatures.append(curvature)
        curvatures = np.array(curvatures)
        # Select 70% high-curvature points, 30% uniform
        n_curvature = int(n_points * 0.7)
        curvature_indices = np.argsort(curvatures)[-n_curvature:]
        uniform_indices = np.random.choice(len(points), n_points - n_curvature, replace=False)
        indices = np.concatenate([curvature_indices, uniform_indices])
        np.random.shuffle(indices)  # Avoid bias in order
        return points[indices]
    except Exception as e:
        logger.warning(f"Curvature-aware sampling failed: {e}. Falling back to random sampling.")
        indices = np.random.choice(len(points), n_points, replace=False)
        return points[indices]

class JawTeethDataset(Dataset):
    def __init__(
        self,
        data_dir,
        max_stages=25,
        num_teeth=14,
        num_points=1000,
        channels=3,
        split='train',
        train_ratio=0.8,
        inference=False,
        cache_dir='./cache',
        log_file='dataset_log.txt',
        use_scaler=True,
        scaler_type='robust'
    ):
        self.data_dir = data_dir
        self.max_stages = max_stages
        self.num_teeth = num_teeth
        self.num_points = num_points
        self.channels = channels
        self.split = split
        self.train_ratio = train_ratio
        self.inference = inference
        self.cache_dir = cache_dir
        self.logger = setup_logging(log_file)
        self.use_scaler = use_scaler
        self.scaler_type = scaler_type.lower()
        self.FDI_TO_INDEX = {
            "31": 0, "32": 1, "33": 2, "34": 3, "35": 4, "36": 5, "37": 6,
            "41": 7, "42": 8, "43": 9, "44": 10, "45": 11, "46": 12, "47": 13
        }
        self.scalers = [None for _ in range(6)]
        if self.use_scaler:
            if self.scaler_type == 'robust':
                self.scalers = [RobustScaler() for _ in range(6)]
            elif self.scaler_type == 'standard':
                self.scalers = [StandardScaler() for _ in range(6)]
            else:
                raise ValueError(f"Invalid scaler_type: {self.scaler_type}. Must be 'robust' or 'standard'.")

        if not self.inference and self.split not in ['train', 'val']:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train' or 'val' for training/validation.")

        os.makedirs(self.cache_dir, exist_ok=True)
        self._initialize_dataset()

    def _preprocess_excel(self, df, jaw_id, is_cumulative=False):
        if df is None or df.empty:
            self.logger.warning(f"No data for Jaw_ID {jaw_id} in Excel; DataFrame is None or empty")
            return None

        self.logger.debug(f"Processing Excel for Jaw_ID {jaw_id}, initial rows: {len(df)}, columns: {df.columns.tolist()}")
        
        df = df[df["Jaw_ID"] == jaw_id].copy()
        self.logger.debug(f"After Jaw_ID filter for {jaw_id}, rows: {len(df)}")
        if df.empty:
            self.logger.warning(f"No data for Jaw_ID {jaw_id} in Excel after filtering by Jaw_ID; unique Jaw_IDs in data: {df['Jaw_ID'].unique().tolist()}")
            return None

        df["Tooth_ID"] = df["Tooth_ID"].astype(str).str.strip().str.replace(',', '.').str.split('.').str[0].str.extract(r'(\d+)')
        self.logger.debug(f"After Tooth_ID processing for Jaw_ID {jaw_id}, unique Tooth_IDs: {df['Tooth_ID'].unique().tolist()}")
        
        df = df[df["Tooth_ID"].isin(self.FDI_TO_INDEX.keys())]
        self.logger.debug(f"After Tooth_ID filter for Jaw_ID {jaw_id}, rows: {len(df)}, valid Tooth_IDs: {df['Tooth_ID'].unique().tolist()}")
        if df.empty:
            self.logger.warning(f"No data for Jaw_ID {jaw_id} in Excel after filtering by Tooth_ID; valid FDI keys: {list(self.FDI_TO_INDEX.keys())}")
            return None

        columns = [
            "Left/Right (mm",
            "Forward/Backward (mm)",
            "Extrude/Intrude (mm)",
            "Buccal/Lingual (degrees)",
            "Mesial/Distal (degrees)",
            "Rotation (degrees)"
        ]
        for col in columns:
            df[col] = df[col].apply(clean_numeric)
            if is_cumulative:
                df[col] = df[col].replace([float('inf'), -float('inf')], 0.0)
            else:
                df[col] = df[col].abs()
                df[col] = df[col].replace([float('inf'), -float('inf')], 0.0)
            if (df[col] < 0).any() and not is_cumulative:
                self.logger.warning(f"Negative values detected in {col} for Jaw_ID {jaw_id} after abs(): {(df[col] < 0).sum()} instances")
        self.logger.debug(f"Processed transformations for Jaw_ID {jaw_id}, is_cumulative={is_cumulative}")

        return df

    def _load_transformations(self, jaw_id, transform_df, num_stages_df, cumulative_df):
        transformations = torch.zeros(self.max_stages, self.num_teeth, 6)
        ratios = torch.zeros(self.max_stages, self.num_teeth, 6)
        directions = torch.ones(self.max_stages, self.num_teeth, 6)
        cumulative_transformations = torch.zeros(self.num_teeth, 6)

        num_stages_data = num_stages_df[num_stages_df["Jaw_ID"] == jaw_id]
        num_stages = min(int(clean_numeric(num_stages_data["Num_Stages"].iloc[0])), self.max_stages) if not num_stages_data.empty else self.max_stages

        if transform_df is not None:
            for _, row in transform_df.iterrows():
                tooth_idx = self.FDI_TO_INDEX[row["Tooth_ID"]]
                stage = int(row["Stage"]) - 1
                if 0 <= stage < self.max_stages:
                    stage_values = torch.tensor([
                        row["Left/Right (mm"],
                        row["Forward/Backward (mm)"],
                        row["Extrude/Intrude (mm)"],
                        row["Buccal/Lingual (degrees)"],
                        row["Mesial/Distal (degrees)"],
                        row["Rotation (degrees)"]
                    ], dtype=torch.float32)
                    transformations[stage, tooth_idx] = stage_values
                    directions[stage, tooth_idx] = torch.tensor([1.0 if x >= 0 else 0.0 for x in stage_values], dtype=torch.float32)

        if cumulative_df is not None:
            for _, row in cumulative_df.iterrows():
                tooth_idx = self.FDI_TO_INDEX[row["Tooth_ID"]]
                cumulative_transformations[tooth_idx] = torch.tensor([
                    row["Left/Right (mm"],
                    row["Forward/Backward (mm)"],
                    row["Extrude/Intrude (mm)"],
                    row["Buccal/Lingual (degrees)"],
                    row["Mesial/Distal (degrees)"],
                    row["Rotation (degrees)"]
                ], dtype=torch.float32)

        zero_net_cases = [
            {'jaw_id': '076', 'tooth': '31', 'param_idx': 0},
            {'jaw_id': '130', 'tooth': '32', 'param_idx': 1},
            {'jaw_id': '226', 'tooth': '32', 'param_idx': 1}
        ]
        for case in zero_net_cases:
            if jaw_id == case['jaw_id'] and case['tooth'] in self.FDI_TO_INDEX:
                tooth_idx = self.FDI_TO_INDEX[case['tooth']]
                param_idx = case['param_idx']
                if abs(cumulative_transformations[tooth_idx, param_idx]) < 1e-6:
                    transformations[:num_stages, tooth_idx, param_idx] = 0.0
                    ratios[:num_stages, tooth_idx, param_idx] = 0.0
                    self.logger.warning(f"Applied Solution 1 for Jaw_ID {jaw_id}, Tooth {case['tooth']}, Parameter {param_idx}: Set transformations and ratios to 0")

        total_abs_movement = torch.sum(torch.abs(transformations[:num_stages]), dim=0)
        for t in range(self.num_teeth):
            for p in range(6):
                if total_abs_movement[t, p] > 1e-6:
                    ratios[:num_stages, t, p] = torch.abs(transformations[:num_stages, t, p]) / total_abs_movement[t, p]
                    sum_ratios = ratios[:num_stages, t, p].sum()
                    if abs(sum_ratios - 1.0) > 1e-4 and sum_ratios > 0:
                        self.logger.warning(f"Sum of ratios for Jaw_ID {jaw_id}, Tooth {t}, Param {p} is {sum_ratios:.4f}, normalizing to 1")
                        ratios[:num_stages, t, p] /= sum_ratios
                else:
                    ratios[:num_stages, t, p] = 0.0

        if self.use_scaler and not self.inference:
            for p in range(6):
                if self.scalers[p] is not None:
                    try:
                        data = cumulative_transformations[:, p].unsqueeze(1).numpy()
                        scaled_data = self.scalers[p].transform(data)
                        cumulative_transformations[:, p] = torch.tensor(scaled_data.flatten(), dtype=torch.float32)
                    except Exception as e:
                        self.logger.error(f"Failed to scale parameter {p} for Jaw_ID {jaw_id}: {e}")

        return transformations, ratios, cumulative_transformations, num_stages, directions

    def _load_cumulative_only(self, jaw_id, cumulative_df):
        cumulative_transformations = torch.zeros(self.num_teeth, 6)
        if cumulative_df is not None:
            for _, row in cumulative_df.iterrows():
                tooth_idx = self.FDI_TO_INDEX[row["Tooth_ID"]]
                cumulative_transformations[tooth_idx] = torch.tensor([
                    row["Left/Right (mm"],
                    row["Forward/Backward (mm)"],
                    row["Extrude/Intrude (mm)"],
                    row["Buccal/Lingual (degrees)"],
                    row["Mesial/Distal (degrees)"],
                    row["Rotation (degrees)"]
                ], dtype=torch.float32)

        if self.inference and self.use_scaler:
            for p in range(6):
                if self.scalers[p] is not None:
                    try:
                        data = cumulative_transformations[:, p].unsqueeze(1).numpy()
                        inverse_scaled_data = self.scalers[p].inverse_transform(data)
                        cumulative_transformations[:, p] = torch.tensor(inverse_scaled_data.flatten(), dtype=torch.float32)
                    except Exception as e:
                        self.logger.error(f"Failed to inverse scale parameter {p} for Jaw_ID {jaw_id}: {e}")

        return cumulative_transformations

    def _preprocess_json(self, json_file, jaw_id):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load JSON file {json_file}: {e}")
            return None if self.inference else (None, None, None)

        feats_list = [None] * self.num_teeth
        vertices_list = [None] * self.num_teeth
        faces_list = [None] * self.num_teeth
        teeth_data = data.get("teeth", {})

        for fdi, tooth_idx in self.FDI_TO_INDEX.items():
            if fdi not in teeth_data:
                self.logger.warning(f"Tooth {fdi} missing in JSON file {json_file}")
                feats = torch.zeros(self.num_points, self.channels, dtype=torch.float32)
                vertices = np.zeros((self.num_points, 3), dtype=np.float32)
                faces = np.zeros((0, 3), dtype=np.int64)
            else:
                tooth_data = teeth_data[fdi]
                vertices = np.array(tooth_data.get("v", []), dtype=np.float32)
                faces = np.array(tooth_data.get("f", []), dtype=np.int64) if "f" in tooth_data else np.zeros((0, 3), dtype=np.int64)

                if len(vertices) == 0:
                    self.logger.warning(f"Tooth {fdi} has no vertices in JSON file {json_file}")
                    feats = torch.zeros(self.num_points, self.channels, dtype=torch.float32)
                    vertices = np.zeros((self.num_points, 3), dtype=np.float32)
                    faces = np.zeros((0, 3), dtype=np.int64)
                else:
                    # Replaced random sampling with curvature-aware sampling
                    if len(vertices) > self.num_points:
                        points = curvature_aware_sampling(vertices, self.num_points, self.logger)
                        # Update faces to match sampled points
                        vertex_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(np.argsort(vertices, axis=0)[:, 0].argsort()[points[:, 0].argsort()])}
                        faces = np.array([[vertex_mapping.get(idx, 0) for idx in face]
                                        for face in faces if all(idx in vertex_mapping for idx in face)], dtype=np.int64)
                    else:
                        points = vertices
                        if len(points) < self.num_points:
                            points = np.pad(points, ((0, self.num_points - len(points)), (0, 0)), mode='constant')[:self.num_points]
                    
                    feats = torch.zeros(self.num_points, self.channels, dtype=torch.float32)
                    feats[:, :3] = torch.tensor(points, dtype=torch.float32)

            feats_list[tooth_idx] = feats
            vertices_list[tooth_idx] = vertices
            faces_list[tooth_idx] = faces

        feats = torch.stack(feats_list)
        return feats if self.inference else (feats, vertices_list, faces_list)

    def _initialize_dataset(self):
        self.logger.info(f"Initializing dataset for split '{self.split}' with use_scaler={self.use_scaler}, scaler_type={self.scaler_type}, inference={self.inference}")

        cases = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d)) and d.isdigit()]
        self.logger.info(f"Found {len(cases)} cases: {cases}")

        if not self.inference:
            train_cases, val_cases = train_test_split(cases, train_size=self.train_ratio, random_state=42)
            self.cases = train_cases if self.split == 'train' else val_cases
        else:
            self.cases = cases  # Use all cases for inference
        self.logger.info(f"Selected {len(self.cases)} cases for split '{self.split}': {self.cases}")

        num_stages_file = os.path.join(self.data_dir, "num_stages.xlsx")
        num_stages_df = pd.read_excel(num_stages_file, dtype={"Jaw_ID": str}) if not self.inference and os.path.exists(num_stages_file) else pd.DataFrame()

        if not self.inference and self.split == 'train' and self.use_scaler:
            all_cumulative_transforms = [[] for _ in range(6)]
            columns = [
                "Left/Right (mm",
                "Forward/Backward (mm)",
                "Extrude/Intrude (mm)",
                "Buccal/Lingual (degrees)",
                "Mesial/Distal (degrees)",
                "Rotation (degrees)"
            ]
            for case in train_cases:
                cumulative_file = os.path.join(self.data_dir, case, "cumulative_transformations.xlsx")
                if os.path.exists(cumulative_file):
                    try:
                        self.logger.debug(f"Loading cumulative_transformations.xlsx for case {case}: {cumulative_file}")
                        df = pd.read_excel(cumulative_file, dtype={"Jaw_ID": str, "Tooth_ID": str})
                        df = self._preprocess_excel(df, case, is_cumulative=True)
                        if df is not None and not df.empty:
                            for i, col in enumerate(columns):
                                all_cumulative_transforms[i].append(df[[col]].values)
                        else:
                            self.logger.warning(f"No valid data after preprocessing cumulative_transformations.xlsx for case {case}")
                    except Exception as e:
                        self.logger.error(f"Failed to load cumulative_transformations.xlsx for case {case}: {e}")
                else:
                    self.logger.warning(f"cumulative_transformations.xlsx not found for case {case}")
            for i in range(6):
                if all_cumulative_transforms[i]:
                    data = np.concatenate(all_cumulative_transforms[i], axis=0)
                    self.scalers[i].fit(data)
                    scaler_file = os.path.join(self.cache_dir, f'scaler_param_{i}.pkl')
                    try:
                        with open(scaler_file, 'wb') as f:
                            pickle.dump(self.scalers[i], f)
                        self.logger.info(f"Saved scaler for parameter {columns[i]} to {scaler_file}")
                    except Exception as e:
                        self.logger.error(f"Failed to save scaler for parameter {columns[i]}: {e}")
                        self.scalers[i] = None
                else:
                    self.logger.warning(f"No valid transformation data for parameter {columns[i]} to fit scaler")
                    self.scalers[i] = RobustScaler() if self.scaler_type == 'robust' else StandardScaler()
        else:
            for i in range(6):
                scaler_file = os.path.join(self.cache_dir, f'scaler_param_{i}.pkl')
                self.logger.debug(f"Checking scaler file: {scaler_file}")
                if os.path.exists(scaler_file) and self.use_scaler:
                    try:
                        with open(scaler_file, 'rb') as f:
                            self.scalers[i] = pickle.load(f)
                        self.logger.info(f"Loaded scaler for parameter {i} from {scaler_file}")
                    except Exception as e:
                        self.logger.error(f"Failed to load scaler for parameter {i} from {scaler_file}: {e}")
                        self.scalers[i] = RobustScaler() if self.scaler_type == 'robust' else StandardScaler()
                else:
                    if self.use_scaler:
                        self.logger.warning(f"No scaler file found for parameter {i} at {scaler_file}; using default scaler")
                        self.scalers[i] = RobustScaler() if self.scaler_type == 'robust' else StandardScaler()

        self.data = []

        for case in self.cases:
            cache_file = os.path.join(self.cache_dir, f"{case}_{self.split}_cache.pkl")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        case_data = pickle.load(f)
                    self.data.append(case_data)
                    self.logger.info(f"Loaded cached data for case {case}")
                    continue
                except Exception as e:
                    self.logger.warning(f"Failed to load cache for case {case}: {e}")

            json_file = os.path.join(self.data_dir, case, "ori", "before_treatment.json")
            transform_file = os.path.join(self.data_dir, case, "Transformations.xlsx")
            cumulative_file = os.path.join(self.data_dir, case, "cumulative_transformations.xlsx")

            if not os.path.exists(json_file):
                self.logger.warning(f"Skipping case {case}: Missing JSON file {json_file}")
                continue

            if self.inference:
                cumulative_df = None
                try:
                    self.logger.debug(f"Checking for cumulative file: {cumulative_file}")
                    if os.path.exists(cumulative_file):
                        cumulative_df = pd.read_excel(cumulative_file, dtype={"Jaw_ID": str, "Tooth_ID": str})
                        self.logger.debug(f"Loaded cumulative_transformations.xlsx for case {case}, rows: {len(cumulative_df)}")
                    else:
                        self.logger.warning(f"cumulative_transformations.xlsx not found for case {case}")
                except Exception as e:
                    self.logger.error(f"Failed to load cumulative_transformations.xlsx for case {case}: {e}")
                cumulative_data = self._preprocess_excel(cumulative_df, case, is_cumulative=True) if cumulative_df is not None else None
                cumulative_transformations = self._load_cumulative_only(case, cumulative_data) if cumulative_data is not None else torch.zeros(self.num_teeth, 6)
                feats = self._preprocess_json(json_file, case)
                if feats is None:
                    self.logger.warning(f"Skipping case {case}: Invalid JSON data")
                    continue
                case_data = {
                    'jaw_id': case,
                    'feats': feats,
                    'cumulative_transformations': cumulative_transformations
                }
            else:
                transform_df = None
                cumulative_df = None
                try:
                    self.logger.debug(f"Checking for transform file: {transform_file}")
                    if os.path.exists(transform_file):
                        transform_df = pd.read_excel(transform_file, dtype={"Jaw_ID": str, "Tooth_ID": str})
                        self.logger.debug(f"Loaded Transformations.xlsx for case {case}, rows: {len(transform_df)}")
                    else:
                        self.logger.warning(f"Transformations.xlsx not found for case {case}")
                    self.logger.debug(f"Checking for cumulative file: {cumulative_file}")
                    if os.path.exists(cumulative_file):
                        cumulative_df = pd.read_excel(cumulative_file, dtype={"Jaw_ID": str, "Tooth_ID": str})
                        self.logger.debug(f"Loaded cumulative_transformations.xlsx for case {case}, rows: {len(cumulative_df)}")
                    else:
                        self.logger.warning(f"cumulative_transformations.xlsx not found for case {case}")
                except Exception as e:
                    self.logger.error(f"Failed to load Excel files for case {case}: {e}")

                transform_data = self._preprocess_excel(transform_df, case) if transform_df is not None else None
                cumulative_data = self._preprocess_excel(cumulative_df, case, is_cumulative=True) if cumulative_df is not None else None
                transformations, ratios, cumulative_transformations, num_stages, directions = self._load_transformations(
                    case, transform_data, num_stages_df, cumulative_data
                )
                feats, _, _ = self._preprocess_json(json_file, case)
                if feats is None:
                    self.logger.warning(f"Skipping case {case}: Invalid JSON data")
                    continue
                case_data = {
                    'jaw_id': case,
                    'feats': feats,
                    'ratios': ratios,
                    'cumulative_transformations': cumulative_transformations,
                    'directions': directions,
                    'num_stages': num_stages
                }

            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(case_data, f)
                self.logger.info(f"Processed and cached data for case {case}")
            except Exception as e:
                self.logger.error(f"Failed to cache data for case {case}: {e}")
            self.data.append(case_data)

        self.logger.info(f"Dataset initialized with {len(self.data)} cases")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.inference:
            return (
                data['jaw_id'],
                data['feats'],
                data['cumulative_transformations']
            )
        
        return (
            data['jaw_id'],
            data['feats'],
            data['ratios'],
            data['cumulative_transformations'],
            data['directions'],
            data['num_stages']
        )

    def get_scalers(self):
        """Return the parameter-specific scalers for external use (e.g., inference)."""
        return self.scalers

class CumulativeJawTeethDataset(Dataset):
    def __init__(
        self,
        data_dir,
        num_teeth=14,
        num_points=256,
        channels=3,
        split='train',
        train_ratio=0.8,
        inference=False,
        cache_dir='./cache',
        log_file='dataset_log.txt',
        use_scaler=True,
        scaler_type='robust'
    ):
        self.data_dir = data_dir
        self.num_teeth = num_teeth
        self.num_points = num_points
        self.channels = channels
        self.split = split
        self.train_ratio = train_ratio
        self.inference = inference
        self.cache_dir = cache_dir
        self.logger = setup_logging(log_file)
        self.use_scaler = use_scaler
        self.scaler_type = scaler_type.lower()
        self.FDI_TO_INDEX = {
            "31": 0, "32": 1, "33": 2, "34": 3, "35": 4, "36": 5, "37": 6,
            "41": 7, "42": 8, "43": 9, "44": 10, "45": 11, "46": 12, "47": 13
        }
        self.scalers = [None] * 6  # Initialize list for 6 scalers
        if self.use_scaler:
            if self.scaler_type == 'robust':
                self.scalers = [RobustScaler() for _ in range(6)]
            elif self.scaler_type == 'standard':
                self.scalers = [StandardScaler() for _ in range(6)]
            else:
                raise ValueError(f"Invalid scaler_type: {self.scaler_type}. Must be 'robust' or 'standard'.")

        if not self.inference and self.split not in ['train', 'val']:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train' or 'val' for training/validation.")

        os.makedirs(self.cache_dir, exist_ok=True)
        self._initialize_dataset()

    def _preprocess_excel(self, df, jaw_id, skip_scaling=False):
        if df is None or df.empty:
            self.logger.warning(f"No data for Jaw_ID {jaw_id} in Excel; DataFrame is None or empty")
            return None

        self.logger.debug(f"Processing Excel for Jaw_ID {jaw_id}, initial rows: {len(df)}, columns: {df.columns.tolist()}")
        
        df = df[df["Jaw_ID"] == jaw_id].copy()
        self.logger.debug(f"After Jaw_ID filter for {jaw_id}, rows: {len(df)}")
        if df.empty:
            self.logger.warning(f"No data for Jaw_ID {jaw_id} in Excel after filtering by Jaw_ID; unique Jaw_IDs in data: {df['Jaw_ID'].unique().tolist()}")
            return None

        df["Tooth_ID"] = df["Tooth_ID"].astype(str).str.strip().str.replace(',', '.').str.split('.').str[0].str.extract(r'(\d+)')
        self.logger.debug(f"After Tooth_ID processing for Jaw_ID {jaw_id}, unique Tooth_IDs: {df['Tooth_ID'].unique().tolist()}")
        
        df = df[df["Tooth_ID"].isin(self.FDI_TO_INDEX.keys())]
        self.logger.debug(f"After Tooth_ID filter for Jaw_ID {jaw_id}, rows: {len(df)}, valid Tooth_IDs: {df['Tooth_ID'].unique().tolist()}")
        if df.empty:
            self.logger.warning(f"No data for Jaw_ID {jaw_id} in Excel after filtering by Tooth_ID; valid FDI keys: {list(self.FDI_TO_INDEX.keys())}")
            return None

        columns = [
            "Left/Right (mm",
            "Forward/Backward (mm)",
            "Extrude/Intrude (mm)",
            "Buccal/Lingual (degrees)",
            "Mesial/Distal (degrees)",
            "Rotation (degrees)"
        ]
        for col in columns:
            df[col] = df[col].apply(clean_numeric)
            df[col] = df[col].replace([float('inf'), -float('inf')], 0.0)

        return df

    def _load_transformations(self, jaw_id, cumulative_df):
        cumulative_transformations = torch.zeros(self.num_teeth, 6)
        # Initialize directions with ones (default positive) for training
        directions = torch.ones(self.num_teeth, 6, dtype=torch.float32) if not self.inference else None

        if cumulative_df is not None:
            for _, row in cumulative_df.iterrows():
                tooth_idx = self.FDI_TO_INDEX[row["Tooth_ID"]]
                values = torch.tensor([
                    row["Left/Right (mm"],
                    row["Forward/Backward (mm)"],
                    row["Extrude/Intrude (mm)"],
                    row["Buccal/Lingual (degrees)"],
                    row["Mesial/Distal (degrees)"],
                    row["Rotation (degrees)"]
                ], dtype=torch.float32)
                cumulative_transformations[tooth_idx] = values
                # Compute directions (0 for negative, 1 for positive) before scaling, only for training
                if not self.inference:
                    directions[tooth_idx] = torch.tensor([1.0 if x >= 0 else 0.0 for x in values], dtype=torch.float32)

        # Compute cumulative_activity and cumulative_param_activity before scaling
        activity_threshold = 1e-6
        cumulative_param_activity = (torch.abs(cumulative_transformations) > activity_threshold).float()  # Shape: (num_teeth, 6)
        cumulative_activity = (cumulative_param_activity.sum(dim=1) > 0).float()  # Shape: (num_teeth)
        if cumulative_activity.sum() == 0:
            self.logger.warning(f"No active teeth detected for Jaw_ID {jaw_id}; all cumulative_activity values are 0")

        if self.use_scaler and not self.inference:
            for p in range(6):
                if self.scalers[p] is not None:
                    try:
                        data = cumulative_transformations[:, p].unsqueeze(1).numpy()
                        scaled_data = self.scalers[p].transform(data)
                        cumulative_transformations[:, p] = torch.tensor(scaled_data.flatten(), dtype=torch.float32)
                    except Exception as e:
                        self.logger.error(f"Failed to scale parameter {p} for Jaw_ID {jaw_id}: {e}")

        if self.inference and self.use_scaler:
            for p in range(6):
                if self.scalers[p] is not None:
                    try:
                        data = cumulative_transformations[:, p].unsqueeze(1).numpy()
                        inverse_scaled_data = self.scalers[p].inverse_transform(data)
                        cumulative_transformations[:, p] = torch.tensor(inverse_scaled_data.flatten(), dtype=torch.float32)
                    except Exception as e:
                        self.logger.error(f"Failed to inverse scale parameter {p} for Jaw_ID {jaw_id}: {e}")

        return cumulative_transformations, cumulative_activity, cumulative_param_activity, directions

    def _preprocess_json(self, json_file, jaw_id):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load JSON file {json_file}: {e}")
            return None

        feats_list = [None] * self.num_teeth
        teeth_data = data.get("teeth", {})

        for fdi, tooth_idx in self.FDI_TO_INDEX.items():
            if fdi not in teeth_data:
                self.logger.warning(f"Tooth {fdi} missing in JSON file {json_file}")
                feats = torch.zeros(self.num_points, self.channels, dtype=torch.float32)
            else:
                tooth_data = teeth_data[fdi]
                vertices = np.array(tooth_data.get("v", []), dtype=np.float32)

                if len(vertices) == 0:
                    self.logger.warning(f"Tooth {fdi} has no vertices in JSON file {json_file}")
                    feats = torch.zeros(self.num_points, self.channels, dtype=torch.float32)
                else:
                    if len(vertices) > self.num_points:
                        points = curvature_aware_sampling(vertices, self.num_points, self.logger)
                    else:
                        points = vertices
                        if len(points) < self.num_points:
                            points = np.pad(points, ((0, self.num_points - len(points)), (0, 0)), mode='constant')[:self.num_points]
                    
                    feats = torch.zeros(self.num_points, self.channels, dtype=torch.float32)
                    feats[:, :3] = torch.tensor(points, dtype=torch.float32)

            feats_list[tooth_idx] = feats

        return torch.stack(feats_list)

    def _initialize_dataset(self):
        self.logger.info(f"Initializing cumulative dataset for split '{self.split}' with use_scaler={self.use_scaler}, scaler_type={self.scaler_type}, inference={self.inference}")

        cases = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d)) and d.isdigit()]
        self.logger.info(f"Found {len(cases)} cases: {cases}")

        if not self.inference:
            train_cases, val_cases = train_test_split(cases, train_size=self.train_ratio, random_state=42)
            self.cases = train_cases if self.split == 'train' else val_cases
        else:
            self.cases = cases
        self.logger.info(f"Selected {len(self.cases)} cases for split '{self.split}': {self.cases}")

        columns = [
            "Left/Right (mm",
            "Forward/Backward (mm)",
            "Extrude/Intrude (mm)",
            "Buccal/Lingual (degrees)",
            "Mesial/Distal (degrees)",
            "Rotation (degrees)"
        ]
        if not self.inference and self.split == 'train' and self.use_scaler:
            all_transforms = [[] for _ in range(6)]
            for case in self.cases:
                cumulative_file = os.path.join(self.data_dir, case, "cumulative_transformations.xlsx")
                if os.path.exists(cumulative_file):
                    try:
                        self.logger.debug(f"Loading cumulative_transformations.xlsx for case {case}: {cumulative_file}")
                        df = pd.read_excel(cumulative_file, dtype={"Jaw_ID": str, "Tooth_ID": str})
                        self.logger.debug(f"Loaded cumulative_transformations.xlsx for case {case}, rows: {len(df)}")
                        df = self._preprocess_excel(df, case)
                        if df is not None and not df.empty:
                            for i, col in enumerate(columns):
                                all_transforms[i].append(df[[col]].values)
                        else:
                            self.logger.warning(f"No valid data after preprocessing cumulative_transformations.xlsx for case {case}")
                    except Exception as e:
                        self.logger.error(f"Failed to load cumulative_transformations.xlsx for case {case}: {e}")
                else:
                    self.logger.warning(f"cumulative_transformations.xlsx not found for case {case}: {cumulative_file}")
            for i in range(6):
                if all_transforms[i]:
                    data = np.concatenate(all_transforms[i], axis=0)
                    try:
                        self.scalers[i].fit(data)
                        scaler_file = os.path.join(self.cache_dir, f'scaler_param_{i}.pkl')
                        with open(scaler_file, 'wb') as f:
                            pickle.dump(self.scalers[i], f)
                        self.logger.info(f"Saved scaler for parameter {columns[i]} to {scaler_file}")
                    except Exception as e:
                        self.logger.error(f"Failed to fit or save scaler for parameter {columns[i]}: {e}")
                        self.scalers[i] = RobustScaler() if self.scaler_type == 'robust' else StandardScaler()
                else:
                    self.logger.warning(f"No valid transformation data for parameter {columns[i]} to fit scaler")
                    self.scalers[i] = RobustScaler() if self.scaler_type == 'robust' else StandardScaler()
        else:
            for i in range(6):
                scaler_file = os.path.join(self.cache_dir, f'scaler_param_{i}.pkl')
                self.logger.debug(f"Checking scaler file: {scaler_file}")
                if os.path.exists(scaler_file) and self.use_scaler:
                    try:
                        with open(scaler_file, 'rb') as f:
                            self.scalers[i] = pickle.load(f)
                        self.logger.info(f"Loaded scaler for parameter {i} from {scaler_file}")
                    except Exception as e:
                        self.logger.error(f"Failed to load scaler for parameter {i} from {scaler_file}: {e}")
                        self.scalers[i] = RobustScaler() if self.scaler_type == 'robust' else StandardScaler()
                else:
                    if self.use_scaler:
                        self.logger.warning(f"No scaler file found for parameter {i} at {scaler_file}; using default scaler")
                        self.scalers[i] = RobustScaler() if self.scaler_type == 'robust' else StandardScaler()

        self.data = []

        for case in self.cases:
            cache_file = os.path.join(self.cache_dir, f"{case}_{self.split}_cumulative_cache.pkl")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        case_data = pickle.load(f)
                    self.data.append(case_data)
                    self.logger.info(f"Loaded cached data for case {case}")
                    continue
                except Exception as e:
                    self.logger.warning(f"Failed to load cache for case {case}: {e}")

            json_file = os.path.join(self.data_dir, case, "ori", "before_treatment.json")
            cumulative_file = os.path.join(self.data_dir, case, "cumulative_transformations.xlsx")

            if not os.path.exists(json_file):
                self.logger.warning(f"Skipping case {case}: Missing JSON file {json_file}")
                continue

            cumulative_df = None
            if not self.inference:
                try:
                    self.logger.debug(f"Checking for cumulative file: {cumulative_file}")
                    if os.path.exists(cumulative_file):
                        cumulative_df = pd.read_excel(cumulative_file, dtype={"Jaw_ID": str, "Tooth_ID": str})
                        self.logger.debug(f"Loaded cumulative_transformations.xlsx for case {case}, rows: {len(cumulative_df)}")
                    else:
                        self.logger.warning(f"cumulative_transformations.xlsx not found for case {case}")
                except Exception as e:
                    self.logger.error(f"Failed to load cumulative_transformations.xlsx for case {case}: {e}")

            cumulative_data = self._preprocess_excel(cumulative_df, case) if not self.inference else None
            cumulative_transformations, cumulative_activity, cumulative_param_activity, directions = self._load_transformations(
                case, cumulative_data
            ) if not self.inference else (torch.zeros(self.num_teeth, 6), torch.zeros(self.num_teeth), torch.zeros(self.num_teeth, 6), None)

            feats = self._preprocess_json(json_file, case)
            if feats is None:
                self.logger.warning(f"Skipping case {case}: Invalid JSON data")
                continue

            case_data = {
                'jaw_id': case,
                'feats': feats,
                'cumulative_transformations': cumulative_transformations,
                'cumulative_activity': cumulative_activity,
                'cumulative_param_activity': cumulative_param_activity,
            }
            if not self.inference:
                case_data['directions'] = directions

            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(case_data, f)
                self.logger.info(f"Processed and cached data for case {case}")
            except Exception as e:
                self.logger.error(f"Failed to cache data for case {case}: {e}")
            self.data.append(case_data)

        self.logger.info(f"Cumulative dataset initialized with {len(self.data)} cases")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.inference:
            return (
                data['jaw_id'],
                data['feats'],
                data['cumulative_transformations']
            )
        return (
            data['jaw_id'],
            data['feats'],
            data['cumulative_transformations'],
            data['cumulative_activity'],
            data['cumulative_param_activity'],
            data['directions']
        )

    def get_scalers(self):
        """Return the parameter-specific scalers for external use (e.g., inference)."""
        return self.scalers