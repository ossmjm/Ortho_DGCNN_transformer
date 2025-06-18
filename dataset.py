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

class JawTeethDataset(Dataset):
    def __init__(
        self,
        data_dir,
        max_stages=25,
        num_teeth=14,
        num_points=256,
        channels=3,
        split='train',
        train_ratio=0.8,
        inference=False,
        cache_dir='./cache',
        log_file='dataset_log.txt'
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
        self.FDI_TO_INDEX = {
            "31": 0, "32": 1, "33": 2, "34": 3, "35": 4, "36": 5, "37": 6,
            "41": 7, "42": 8, "43": 9, "44": 10, "45": 11, "46": 12, "47": 13
        }
        self.scalers = [RobustScaler() for _ in range(6)]  # For Transformations
        self.cumulative_scalers = [RobustScaler() for _ in range(6)]  # For cumulative_transformations

        if self.split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'val', or 'test'.")

        os.makedirs(self.cache_dir, exist_ok=True)
        self._initialize_dataset()

    def _preprocess_excel(self, df, jaw_id, is_cumulative=False, skip_scaling=False):
        if df is None or df.empty:
            self.logger.warning(f"No data for Jaw_ID {jaw_id} in Excel")
            return None

        df = df[df["Jaw_ID"] == jaw_id].copy()
        if df.empty:
            self.logger.warning(f"No data for Jaw_ID {jaw_id} in Excel after filtering")
            return None

        df["Tooth_ID"] = df["Tooth_ID"].astype(str).str.strip().str.replace(',', '.').str.split('.').str[0].str.extract(r'(\d+)')
        df = df[df["Tooth_ID"].isin(self.FDI_TO_INDEX.keys())]
        if df.empty:
            self.logger.warning(f"No valid Tooth_ID for Jaw_ID {jaw_id} after cleaning")
            return None

        columns = [
            "Left/Right (mm", "Forward/Backward (mm)", "Extrude/Intrude (mm)",
            "Buccal/Lingual (degrees)", "Mesial/Distal (degrees)", "Rotation (degrees)"
        ]
        for col in columns:
            df[col] = df[col].apply(clean_numeric)
            df[col] = df[col].abs()  # Convert to absolute values
            df[col] = df[col].replace([float('inf'), -float('inf')], 0.0)
        self.logger.debug(f"Converted transformation values to absolute for Jaw_ID {jaw_id}")

        if not is_cumulative:
            df["Stage"] = df["Stage"].apply(clean_numeric).astype(int)
            df["Stage"] = df["Stage"].fillna(1).clip(lower=1, upper=self.max_stages)
            df = df[df["Stage"] <= self.max_stages]
            if df["Stage"].isna().any():
                self.logger.error(f"Stage column for Jaw_ID {jaw_id} contains NaN after imputation")
                return None

        if not self.inference and not skip_scaling and len(df) > 0:
            scalers = self.cumulative_scalers if is_cumulative else self.scalers
            for i, col in enumerate(columns):
                data = df[[col]].values
                if self.split == 'train':
                    scaled_data = scalers[i].fit_transform(data)
                else:
                    scaled_data = scalers[i].transform(data) if scalers[i] is not None else data
                df[col] = scaled_data.flatten()

        return df

    def _load_transformations(self, jaw_id, transform_df, cumulative_df, num_stages_df):
        transformations = torch.zeros(self.max_stages, self.num_teeth, 6)
        cumulative_transformations = torch.zeros(self.num_teeth, 6)
        directions = torch.ones(self.num_teeth, 6)  # Default to 1 (positive or zero)
        type_labels = torch.zeros(self.max_stages, self.num_teeth, dtype=torch.long)

        num_stages_data = num_stages_df[num_stages_df["Jaw_ID"] == jaw_id]
        num_stages = min(int(clean_numeric(num_stages_data["Num_Stages"].iloc[0])), self.max_stages) if not num_stages_data.empty else self.max_stages

        if transform_df is not None:
            for _, row in transform_df.iterrows():
                tooth_idx = self.FDI_TO_INDEX[row["Tooth_ID"]]
                stage = int(row["Stage"]) - 1
                transformations[stage, tooth_idx] = torch.tensor([
                    row["Left/Right (mm"], row["Forward/Backward (mm)"], row["Extrude/Intrude (mm)"],
                    row["Buccal/Lingual (degrees)"], row["Mesial/Distal (degrees)"], row["Rotation (degrees)"]
                ], dtype=torch.float32)
                trans = transformations[stage, tooth_idx, :3]
                rot = transformations[stage, tooth_idx, 3:]
                has_trans = torch.any(trans != 0).item()
                has_rot = torch.any(rot != 0).item()
                if not has_trans and not has_rot:
                    type_labels[stage, tooth_idx] = 0
                elif has_trans and not has_rot:
                    type_labels[stage, tooth_idx] = 1
                elif not has_trans and has_rot:
                    type_labels[stage, tooth_idx] = 2
                else:
                    type_labels[stage, tooth_idx] = 3

        if cumulative_df is not None:
            raw_values = []  # Store raw values before taking absolute
            for _, row in cumulative_df.iterrows():
                tooth_idx = self.FDI_TO_INDEX[row["Tooth_ID"]]
                raw = [
                    row["Left/Right (mm"], row["Forward/Backward (mm)"], row["Extrude/Intrude (mm)"],
                    row["Buccal/Lingual (degrees)"], row["Mesial/Distal (degrees)"], row["Rotation (degrees)"]
                ]
                raw_values.append(raw)
                cumulative_transformations[tooth_idx] = torch.tensor([abs(x) for x in raw], dtype=torch.float32)
                directions[tooth_idx] = torch.tensor([1.0 if x >= 0 else 0.0 for x in raw], dtype=torch.float32)
            self.logger.debug(f"Extracted directions for Jaw_ID {jaw_id}: {directions.tolist()}")

        activity = torch.any(transformations != 0, dim=-1).float()
        param_activity = (transformations != 0).float()
        cumulative_activity = torch.any(cumulative_transformations != 0, dim=-1).float()
        cumulative_param_activity = (cumulative_transformations != 0).float()

        return transformations, cumulative_transformations, type_labels, num_stages, activity, param_activity, cumulative_activity, cumulative_param_activity, directions

    def _load_cumulative_only(self, jaw_id, cumulative_df):
        cumulative_transformations = torch.zeros(self.num_teeth, 6)
        directions = torch.ones(self.num_teeth, 6)  # Default to 1
        if cumulative_df is not None:
            for _, row in cumulative_df.iterrows():
                tooth_idx = self.FDI_TO_INDEX[row["Tooth_ID"]]
                raw = [
                    row["Left/Right (mm"], row["Forward/Backward (mm)"], row["Extrude/Intrude (mm)"],
                    row["Buccal/Lingual (degrees)"], row["Mesial/Distal (degrees)"], row["Rotation (degrees)"]
                ]
                cumulative_transformations[tooth_idx] = torch.tensor([abs(x) for x in raw], dtype=torch.float32)
                directions[tooth_idx] = torch.tensor([1.0 if x >= 0 else 0.0 for x in raw], dtype=torch.float32)
        return cumulative_transformations, directions

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
                    np.random.seed(42)
                    if len(vertices) > self.num_points:
                        indices = np.random.choice(len(vertices), self.num_points, replace=False)
                        points = vertices[indices]
                        vertex_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}
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
        self.logger.info(f"Initializing dataset for split '{self.split}'")

        num_stages_file = os.path.join(self.data_dir, "num_stages.xlsx")
        num_stages_df = pd.read_excel(num_stages_file, dtype={"Jaw_ID": str}) if not self.inference and os.path.exists(num_stages_file) else pd.DataFrame()

        cases = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d)) and d.isdigit()]
        self.logger.info(f"Found {len(cases)} cases: {cases}")

        train_val_cases, test_cases = train_test_split(cases, train_size=self.train_ratio, random_state=42)
        train_cases, val_cases = train_test_split(train_val_cases, train_size=self.train_ratio, random_state=42)
        if self.split == 'train':
            self.cases = train_cases
        elif self.split == 'val':
            self.cases = val_cases
        else:
            self.cases = test_cases
        self.logger.info(f"Selected {len(self.cases)} cases for split '{self.split}': {self.cases}")

        if self.split == 'train' and not self.inference:
            all_transforms = [[] for _ in range(6)]
            all_cumulative_transforms = [[] for _ in range(6)]
            columns = [
                "Left/Right (mm", "Forward/Backward (mm)", "Extrude/Intrude (mm)",
                "Buccal/Lingual (degrees)", "Mesial/Distal (degrees)", "Rotation (degrees)"
            ]
            for case in train_cases:
                transform_file = os.path.join(self.data_dir, case, "Transformations.xlsx")
                cumulative_file = os.path.join(self.data_dir, case, "cumulative_transformations.xlsx")
                if os.path.exists(transform_file):
                    try:
                        df = pd.read_excel(transform_file, dtype={"Jaw_ID": str, "Tooth_ID": str})
                        df = self._preprocess_excel(df, case, is_cumulative=False, skip_scaling=True)
                        if df is not None and not df.empty:
                            for i, col in enumerate(columns):
                                all_transforms[i].append(df[[col]].values)
                    except Exception as e:
                        self.logger.error(f"Failed to load Transformations.xlsx for case {case}: {e}")
                if os.path.exists(cumulative_file):
                    try:
                        df = pd.read_excel(cumulative_file, dtype={"Jaw_ID": str, "Tooth_ID": str})
                        df = self._preprocess_excel(df, case, is_cumulative=True, skip_scaling=True)
                        if df is not None and not df.empty:
                            for i, col in enumerate(columns):
                                all_cumulative_transforms[i].append(df[[col]].values)
                    except Exception as e:
                        self.logger.error(f"Failed to load cumulative_transformations.xlsx for case {case}: {e}")
            for i in range(6):
                if all_transforms[i]:
                    data = np.concatenate(all_transforms[i], axis=0)
                    self.scalers[i].fit(data)
                    scaler_file = os.path.join(self.cache_dir, f'scaler_param_{i}.pkl')
                    try:
                        with open(scaler_file, 'wb') as f:
                            pickle.dump(self.scalers[i], f)
                        self.logger.info(f"Saved scaler for parameter {columns[i]} to {scaler_file}")
                    except Exception as e:
                        self.logger.error(f"Failed to save scaler for parameter {columns[i]}: {e}")
                else:
                    self.logger.warning(f"No valid transformation data for parameter {columns[i]} to fit scaler")
                    self.scalers[i] = None
                if all_cumulative_transforms[i]:
                    data = np.concatenate(all_cumulative_transforms[i], axis=0)
                    self.cumulative_scalers[i].fit(data)
                    scaler_file = os.path.join(self.cache_dir, f'cumulative_scaler_param_{i}.pkl')
                    try:
                        with open(scaler_file, 'wb') as f:
                            pickle.dump(self.scalers[i], f)
                        self.logger.info(f"Saved cumulative scaler for parameter {columns[i]} to {scaler_file}")
                    except Exception as e:
                        self.logger.error(f"Failed to save scaler for parameter {columns[i]}: {e}")
                else:
                    self.logger.warning(f"No valid cumulative transformation data for parameter {columns[i]} to fit scaler")
                    self.cumulative_scalers[i] = None
        else:
            for i in range(6):
                scaler_file = os.path.join(self.cache_dir, f'scaler_param_{i}.pkl')
                if os.path.exists(scaler_file):
                    try:
                        with open(scaler_file, 'rb') as f:
                            self.scalers[i] = pickle.load(f)
                        self.logger.info(f"Loaded scaler for parameter {i} from {scaler_file}")
                    except Exception as e:
                        self.logger.error(f"Failed to load scaler for parameter {i}: {e}")
                        self.scalers[i] = None
                else:
                    self.logger.warning(f"No scaler found for parameter {i}; proceeding without scaling")
                    self.scalers[i] = None
                cumulative_scaler_file = os.path.join(self.cache_dir, f'cumulative_scaler_param_{i}.pkl')
                if os.path.exists(cumulative_scaler_file):
                    try:
                        with open(cumulative_scaler_file, 'rb') as f:
                            self.cumulative_scalers[i] = pickle.load(f)
                        self.logger.info(f"Loaded cumulative scaler for parameter {i} from {cumulative_scaler_file}")
                    except Exception as e:
                        self.logger.error(f"Failed to load cumulative scaler for parameter {i}: {e}")
                        self.cumulative_scalers[i] = None
                else:
                    self.logger.warning(f"No cumulative scaler found for parameter {i}; proceeding without scaling")
                    self.cumulative_scalers[i] = None

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
                    cumulative_df = pd.read_excel(cumulative_file, dtype={"Jaw_ID": str, "Tooth_ID": str}) if os.path.exists(cumulative_file) else None
                except Exception as e:
                    self.logger.error(f"Failed to load cumulative_transformations.xlsx for case {case}: {e}")
                cumulative_data = self._preprocess_excel(cumulative_df, case, is_cumulative=True) if cumulative_df is not None else None
                cumulative_transformations, directions = self._load_cumulative_only(case, cumulative_data) if cumulative_data is not None else (torch.zeros(self.num_teeth, 6), torch.ones(self.num_teeth, 6))
                num_stages = self.max_stages
                feats = self._preprocess_json(json_file, case)
                if feats is None:
                    self.logger.warning(f"Skipping case {case}: Invalid JSON data")
                    continue
                case_data = {
                    'jaw_id': case,
                    'feats': feats,
                    'cumulative_transformations': cumulative_transformations,
                    'directions': directions,
                    'num_stages': num_stages
                }
            else:
                transform_df = None
                cumulative_df = None
                try:
                    transform_df = pd.read_excel(transform_file, dtype={"Jaw_ID": str, "Tooth_ID": str}) if os.path.exists(transform_file) else None
                except Exception as e:
                    self.logger.error(f"Failed to load Transformations.xlsx for case {case}: {e}")
                try:
                    cumulative_df = pd.read_excel(cumulative_file, dtype={"Jaw_ID": str, "Tooth_ID": str}) if os.path.exists(cumulative_file) else None
                except Exception as e:
                    self.logger.error(f"Failed to load cumulative_transformations.xlsx for case {case}: {e}")

                transform_data = self._preprocess_excel(transform_df, case) if transform_df is not None else None
                cumulative_data = self._preprocess_excel(cumulative_df, case, is_cumulative=True) if cumulative_df is not None else None
                transformations, cumulative_transformations, type_labels, num_stages, activity, param_activity, cumulative_activity, cumulative_param_activity, directions = self._load_transformations(
                    case, transform_data, cumulative_data, num_stages_df
                )
                feats, vertices, faces = self._preprocess_json(json_file, case)
                if feats is None:
                    self.logger.warning(f"Skipping case {case}: Invalid JSON data")
                    continue
                case_data = {
                    'jaw_id': case,
                    'feats': feats,
                    'transformations': transformations,
                    'cumulative_transformations': cumulative_transformations,
                    'activity': activity,
                    'param_activity': param_activity,
                    'type_labels': type_labels,
                    'cumulative_activity': cumulative_activity,
                    'cumulative_param_activity': cumulative_param_activity,
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
                data['cumulative_transformations'],
                data['directions'],
                data['num_stages']
            )
        return (
            data['jaw_id'],
            data['feats'],
            data['transformations'],
            data['cumulative_transformations'],
            data['activity'],
            data['param_activity'],
            data['type_labels'],
            data['cumulative_activity'],
            data['cumulative_param_activity'],
            data['directions'],
            data['num_stages']
        )

    def get_scalers(self):
        """Return the parameter-specific scalers for external use (e.g., inference)."""
        return self.scalers, self.cumulative_scalers

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
        log_file='dataset_log.txt'
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
        self.FDI_TO_INDEX = {
            "31": 0, "32": 1, "33": 2, "34": 3, "35": 4, "36": 5, "37": 6,
            "41": 7, "42": 8, "43": 9, "44": 10, "45": 11, "46": 12, "47": 13
        }
        self.scaler = None

        if self.split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'val', or 'test'.")

        os.makedirs(self.cache_dir, exist_ok=True)
        self._initialize_dataset()

    def _preprocess_excel(self, df, jaw_id, skip_scaling=False):
        if df is None or df.empty:
            self.logger.warning(f"No data for Jaw_ID {jaw_id} in Excel")
            return None

        df = df[df["Jaw_ID"] == jaw_id].copy()
        if df.empty:
            self.logger.warning(f"No data for Jaw_ID {jaw_id} in Excel after filtering")
            return None

        df["Tooth_ID"] = df["Tooth_ID"].astype(str).str.strip().str.replace(',', '.').str.split('.').str[0].str.extract(r'(\d+)')
        df = df[df["Tooth_ID"].isin(self.FDI_TO_INDEX.keys())]
        if df.empty:
            self.logger.warning(f"No valid Tooth_ID for Jaw_ID {jaw_id} after cleaning")
            return None

        columns = ["Left/Right (mm", "Forward/Backward (mm)", "Extrude/Intrude (mm)",
                   "Buccal/Lingual (degrees)", "Mesial/Distal (degrees)", "Rotation (degrees)"]
        for col in columns:
            df[col] = df[col].apply(clean_numeric)
            df[col] = df[col].replace([float('inf'), -float('inf')], 0)

        if self.scaler is not None and not self.inference and not skip_scaling:
            try:
                data = df[columns]
                scaled_data = self.scaler.transform(data)
                df[columns] = scaled_data
                self.logger.debug(f"Applied StandardScaler to transformations for Jaw_ID {jaw_id}")
            except Exception as e:
                self.logger.error(f"Failed to apply StandardScaler for Jaw_ID {jaw_id}: {e}")
                return None

        return df

    def _load_transformations(self, jaw_id, cumulative_df):
        cumulative_transformations = torch.zeros(self.num_teeth, 6)
        if cumulative_df is not None:
            for _, row in cumulative_df.iterrows():
                tooth_idx = self.FDI_TO_INDEX[row["Tooth_ID"]]
                cumulative_transformations[tooth_idx] = torch.tensor([
                    row["Left/Right (mm"], row["Forward/Backward (mm)"], row["Extrude/Intrude (mm)"],
                    row["Buccal/Lingual (degrees)"], row["Mesial/Distal (degrees)"], row["Rotation (degrees)"]
                ], dtype=torch.float32)

        cumulative_activity = torch.any(cumulative_transformations != 0, dim=-1).float()
        cumulative_param_activity = (cumulative_transformations != 0).float()

        return cumulative_transformations, cumulative_activity, cumulative_param_activity

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

                np.random.seed(42)
                if len(vertices) > self.num_points:
                    indices = np.random.choice(len(vertices), self.num_points, replace=False)
                    points = vertices[indices]
                else:
                    points = vertices
                    if len(points) < self.num_points:
                        points = np.pad(points, ((0, self.num_points - len(points)), (0, 0)), mode='constant')[:self.num_points]
                
                feats = torch.zeros(self.num_points, self.channels, dtype=torch.float32)
                feats[:, :3] = torch.tensor(points, dtype=torch.float32)

            feats_list[tooth_idx] = feats

        return torch.stack(feats_list)

    def _initialize_dataset(self):
        self.logger.info(f"Initializing cumulative dataset for split '{self.split}'")

        cases = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d)) and d.isdigit()]
        self.logger.info(f"Found {len(cases)} cases: {cases}")

        train_val_cases, test_cases = train_test_split(cases, train_size=self.train_ratio, random_state=42)
        train_cases, val_cases = train_test_split(train_val_cases, train_size=self.train_ratio, random_state=42)
        if self.split == 'train':
            self.cases = train_cases
        elif self.split == 'val':
            self.cases = val_cases
        else:
            self.cases = test_cases
        self.logger.info(f"Selected {len(self.cases)} cases for split '{self.split}': {self.cases}")

        scaler_file = os.path.join(self.cache_dir, 'scaler.pkl')
        if self.split == 'train' and not self.inference:
            self.scaler = StandardScaler()
            all_transforms = []
            for case in train_cases:
                cumulative_file = os.path.join(self.data_dir, case, "cumulative_transformations.xlsx")
                if os.path.exists(cumulative_file):
                    try:
                        df = pd.read_excel(cumulative_file, dtype={"Jaw_ID": str, "Tooth_ID": str})
                        df = self._preprocess_excel(df, case, skip_scaling=True)
                        if df is not None and not df.empty:
                            columns = ["Left/Right (mm", "Forward/Backward (mm)", "Extrude/Intrude (mm)",
                                       "Buccal/Lingual (degrees)", "Mesial/Distal (degrees)", "Rotation (degrees)"]
                            all_transforms.append(df[columns])
                    except Exception as e:
                        self.logger.error(f"Failed to load cumulative_transformations.xlsx for case {case}: {e}")
            if all_transforms:
                all_transforms = pd.concat(all_transforms, ignore_index=True)
                self.scaler.fit(all_transforms)
                try:
                    with open(scaler_file, 'wb') as f:
                        pickle.dump(self.scaler, f)
                    self.logger.info(f"Saved StandardScaler to {scaler_file}")
                except Exception as e:
                    self.logger.error(f"Failed to save StandardScaler: {e}")
            else:
                self.logger.warning("No valid transformation data to fit StandardScaler")
                self.scaler = None
        else:
            if os.path.exists(scaler_file):
                try:
                    with open(scaler_file, 'rb') as f:
                        self.scaler = pickle.load(f)
                    self.logger.info(f"Loaded StandardScaler from {scaler_file}")
                except Exception as e:
                    self.logger.error(f"Failed to load StandardScaler: {e}")
                    self.scaler = None
            else:
                self.logger.warning("No StandardScaler found; proceeding without scaling")
                self.scaler = None

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
                    cumulative_df = pd.read_excel(cumulative_file, dtype={"Jaw_ID": str, "Tooth_ID": str}) if os.path.exists(cumulative_file) else None
                except Exception as e:
                    self.logger.error(f"Failed to load cumulative_transformations.xlsx for case {case}: {e}")

            cumulative_data = self._preprocess_excel(cumulative_df, case) if not self.inference else None
            cumulative_transformations, cumulative_activity, cumulative_param_activity = self._load_transformations(
                case, cumulative_data
            ) if not self.inference else (
                torch.zeros(self.num_teeth, 6),
                torch.zeros(self.num_teeth),
                torch.zeros(self.num_teeth, 6)
            )

            feats = self._preprocess_json(json_file, case)
            if feats is None:
                self.logger.warning(f"Skipping case {case}: Invalid JSON data")
                continue

            case_data = {
                'jaw_id': case,
                'feats': feats,
                'cumulative_transformations': cumulative_transformations,
                'cumulative_activity': cumulative_activity,
                'cumulative_param_activity': cumulative_param_activity
            }
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
            data['cumulative_param_activity']
        )