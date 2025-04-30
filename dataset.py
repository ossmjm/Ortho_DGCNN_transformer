import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
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
        channels=13,
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

        if self.split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'val', or 'test'.")

        os.makedirs(self.cache_dir, exist_ok=True)
        self._initialize_dataset()

    def _preprocess_excel(self, df, jaw_id, is_cumulative=False):
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

        if not is_cumulative:
            df["Stage"] = df["Stage"].apply(clean_numeric).astype(int)
            if df["Stage"].isna().any() or (df["Stage"] <= 0).any():
                df["Stage"] = df["Stage"].fillna(method='ffill').fillna(1).astype(int)
            df = df[df["Stage"] <= self.max_stages]
            if df["Stage"].isna().any():
                raise ValueError(f"Stage column for Jaw_ID {jaw_id} contains NaN after imputation")

        return df

    def _load_transformations(self, jaw_id, transform_df, cumulative_df, num_stages_df):
        transformations = torch.zeros(self.max_stages, self.num_teeth, 6)
        cumulative_transformations = torch.zeros(self.num_teeth, 6)
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
            for _, row in cumulative_df.iterrows():
                tooth_idx = self.FDI_TO_INDEX[row["Tooth_ID"]]
                cumulative_transformations[tooth_idx] = torch.tensor([
                    row["Left/Right (mm"], row["Forward/Backward (mm)"], row["Extrude/Intrude (mm)"],
                    row["Buccal/Lingual (degrees)"], row["Mesial/Distal (degrees)"], row["Rotation (degrees)"]
                ], dtype=torch.float32)

        return transformations, cumulative_transformations, type_labels, num_stages

    def _preprocess_json(self, json_file, jaw_id):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load JSON file {json_file}: {e}")
            return None, None, None

        feats_list, vertices_list, faces_list = [None] * self.num_teeth, [None] * self.num_teeth, [None] * self.num_teeth
        teeth_data = data.get("teeth", {})

        for fdi, tooth_idx in self.FDI_TO_INDEX.items():
            if fdi not in teeth_data:
                self.logger.warning(f"Tooth {fdi} missing in JSON file {json_file}")
                feats = torch.zeros(self.num_points, self.channels, dtype=torch.float32)
                feats[:, 12] = tooth_idx
                vertices = np.zeros((self.num_points, 3), dtype=np.float32)
                faces = np.zeros((0, 3), dtype=np.int64)
            else:
                tooth_data = teeth_data[fdi]
                vertices = np.array(tooth_data.get("v", []), dtype=np.float32)
                faces = np.array(tooth_data.get("f", []), dtype=np.int64) if "f" in tooth_data else np.zeros((0, 3), dtype=np.int64)

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
                        points = np.pad(points, ((0, self.num_points - len(points)), (0, 0)), mode='wrap')[:self.num_points]
                
                feats = torch.zeros(self.num_points, self.channels, dtype=torch.float32)
                feats[:, :3] = torch.tensor(points, dtype=torch.float32)
                feats[:, 12] = tooth_idx

            feats_list[tooth_idx] = feats
            vertices_list[tooth_idx] = vertices
            faces_list[tooth_idx] = faces

        return torch.stack(feats_list), vertices_list, faces_list

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
        if self.inference:
            self.cases = cases
        self.logger.info(f"Selected {len(self.cases)} cases for split '{self.split}': {self.cases}")

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

            transform_df = None
            cumulative_df = None
            if not self.inference:
                try:
                    transform_df = pd.read_excel(transform_file, dtype={"Jaw_ID": str, "Tooth_ID": str}) if os.path.exists(transform_file) else None
                except Exception as e:
                    self.logger.error(f"Failed to load Transformations.xlsx for case {case}: {e}")
                try:
                    cumulative_df = pd.read_excel(cumulative_file, dtype={"Jaw_ID": str, "Tooth_ID": str}) if os.path.exists(cumulative_file) else None
                except Exception as e:
                    self.logger.error(f"Failed to load cumulative_transformations.xlsx for case {case}: {e}")

            transform_data = self._preprocess_excel(transform_df, case) if not self.inference else None
            cumulative_data = self._preprocess_excel(cumulative_df, case, is_cumulative=True) if not self.inference else None
            transformations, cumulative_transformations, type_labels, num_stages = self._load_transformations(
                case, transform_data, cumulative_data, num_stages_df
            ) if not self.inference else (
                torch.zeros(self.max_stages, self.num_teeth, 6),
                torch.zeros(self.num_teeth, 6),
                torch.zeros(self.max_stages, self.num_teeth, dtype=torch.long),
                self.max_stages
            )

            activity = torch.any(transformations != 0, dim=-1).float()
            param_activity = (transformations != 0).float()
            cumulative_activity = torch.any(cumulative_transformations != 0, dim=-1).float()
            cumulative_param_activity = (cumulative_transformations != 0).float()
            print(f'cumulative activity: {cumulative_activity}, cumulative_param_activity: {cumulative_param_activity}')
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
                'num_stages': num_stages,
                'vertices': vertices,
                'faces': faces
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
                data['transformations'],
                data['cumulative_transformations'],
                data['vertices'],
                data['faces'],
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
            data['num_stages']
        )