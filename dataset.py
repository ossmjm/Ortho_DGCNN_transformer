import os
import torch
import numpy as np
import pandas as pd
import trimesh
import logging
import json
from torch.utils.data import Dataset

class JawTeethDataset(Dataset):
    def __init__(self, data_dir, max_stages=25, split='train', train_ratio=0.8, inference=False, log_file='log.txt'):
        self.data_dir = data_dir
        self.max_stages = max_stages
        self.split = split
        self.train_ratio = train_ratio
        self.inference = inference
        self.logger = logging.getLogger('TrainLogger' if not inference else 'InferenceLogger')
        if not self.logger.handlers:
            logging.basicConfig(filename=log_file, level=logging.INFO, 
                              format='%(asctime)s - %(levelname)s - %(message)s')
        
        self.jaw_list = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        self.jaw_list.sort()
        self.num_teeth = 14
        self.FDI_numbers = [31, 32, 33, 34, 35, 36, 37, 41, 42, 43, 44, 45, 46, 47]
        
        split_idx = int(len(self.jaw_list) * train_ratio)
        if split == 'train':
            self.jaw_list = self.jaw_list[:split_idx]
        else:
            self.jaw_list = self.jaw_list[split_idx:]
        
        self.data = []
        for jaw in self.jaw_list:
            jaw_path = os.path.join(data_dir, jaw)
            json_file = os.path.join(jaw_path,'ori','before_treatment.json')
            transform_file = os.path.join(jaw_path, 'Transformations.xlsx')
            
            # Check for JSON file
            if not os.path.exists(json_file):
                self.logger.warning(f"JSON file not found for jaw {jaw}")
                continue
            
            # Check for Transformations.xlsx (unless in inference mode)
            if not os.path.exists(transform_file) and not inference:
                self.logger.warning(f"Transformations.xlsx not found for jaw {jaw}")
                continue
            
            num_stages = self.max_stages
            if not inference:
                df = pd.read_excel(transform_file)
                stages = df['Stage'].unique()
                num_stages = min(len(stages), max_stages)
            
            # Load JSON data
            try:
                with open(json_file, 'r') as f:
                    json_data = json.load(f)
                teeth_data = json_data.get('teeth', {})
            except Exception as e:
                self.logger.error(f"Error loading JSON for jaw {jaw}: {e}")
                continue
            
            feats_list = []
            vertices_list = []
            faces_list = []
            for tooth_idx, fdi in enumerate(self.FDI_numbers):
                fdi_str = str(fdi)
                if fdi_str not in teeth_data:
                    self.logger.warning(f"Tooth {fdi} data not found in JSON for jaw {jaw}")
                    feats_list.append(torch.zeros(2048, 13))
                    vertices_list.append(np.zeros((0, 3)))
                    faces_list.append(np.zeros((0, 3), dtype=np.int64))
                    continue
                
                try:
                    tooth_data = teeth_data[fdi_str]
                    vertices = np.array(tooth_data['v'], dtype=np.float32)
                    faces = np.array(tooth_data['f'], dtype=np.int64) if 'f' in tooth_data else np.zeros((0, 3), dtype=np.int64)
                    
                    feats = self.preprocess_tooth_points(vertices, tooth_idx)
                    feats_list.append(feats)
                    vertices_list.append(vertices)
                    faces_list.append(faces)
                except Exception as e:
                    self.logger.error(f"Error processing tooth {fdi} in jaw {jaw}: {e}")
                    feats_list.append(torch.zeros(2048, 13))
                    vertices_list.append(np.zeros((0, 3)))
                    faces_list.append(np.zeros((0, 3), dtype=np.int64))
            
            if not inference:
                targets, activity_labels, param_activity_labels = self.process_transformations(df, num_stages)
                self.data.append({
                    'jaw_id': jaw,
                    'feats_list': feats_list,
                    'targets': targets,
                    'num_stages': num_stages,
                    'activity_labels': activity_labels,
                    'param_activity_labels': param_activity_labels,
                    'vertices_list': vertices_list,
                    'faces_list': faces_list
                })
            else:
                self.data.append({
                    'jaw_id': jaw,
                    'feats_list': feats_list,
                    'targets': torch.zeros(max_stages, self.num_teeth, 6),
                    'num_stages': num_stages,
                    'activity_labels': torch.zeros(max_stages, self.num_teeth),
                    'param_activity_labels': torch.zeros(max_stages, self.num_teeth, 6),
                    'vertices_list': vertices_list,
                    'faces_list': faces_list
                })
        
        self.logger.info(f"Loaded {len(self.data)} jaws for {split} split")
    
    def preprocess_tooth_points(self, vertices, tooth_idx):
        if len(vertices) == 0:
            return torch.zeros(2048, 13)
        
        np.random.seed(42)
        if len(vertices) > 2048:
            indices = np.random.choice(len(vertices), 2048, replace=False)
            points = vertices[indices]
        else:
            points = vertices
            while len(points) < 2048:
                points = np.concatenate([points, vertices[np.random.choice(len(vertices), min(len(vertices), 2048 - len(points)))]])
            points = points[:2048]
        
        centroid = np.mean(points, axis=0)
        points = points - centroid
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms.max() if norms.max() > 0 else points
        
        mesh = trimesh.Trimesh(vertices=points, process=False)
        normals = mesh.vertex_normals
        tangents = np.cross(normals, np.random.randn(*normals.shape))
        tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
        bitangents = np.cross(normals, tangents)
        
        tooth_idx_tensor = np.full((2048, 1), tooth_idx)
        feats = np.concatenate([points, centroid[np.newaxis, :].repeat(2048, axis=0), 
                              normals, tangents, tooth_idx_tensor], axis=1)
        return torch.tensor(feats, dtype=torch.float32)
    
    def process_transformations(self, df, num_stages):
        targets = torch.zeros(self.max_stages, self.num_teeth, 6)
        activity_labels = torch.zeros(self.max_stages, self.num_teeth)
        param_activity_labels = torch.zeros(self.max_stages, self.num_teeth, 6)
        
        FDI_to_idx = {fdi: idx for idx, fdi in enumerate(self.FDI_numbers)}
        
        for stage in range(1, num_stages + 1):
            stage_df = df[df['Stage'] == stage]
            for _, row in stage_df.iterrows():
                tooth_id = row['Tooth_ID']
                if tooth_id not in FDI_to_idx:
                    continue
                tooth_idx = FDI_to_idx[tooth_id]
                transform = [
                    row['Left/Right (mm'], row['Forward/Backward (mm)'], row['Extrude/Intrude (mm)'],
                    row['Buccal/Lingual (degrees)'], row['Mesial/Distal (degrees)'], row['Rotation (degrees)']
                ]
                targets[stage - 1, tooth_idx] = torch.tensor(transform, dtype=torch.float32)
                
                activity = 1.0 if any(abs(x) > 1e-6 for x in transform) else 0.0
                activity_labels[stage - 1, tooth_idx] = activity
                param_activity_labels[stage - 1, tooth_idx] = torch.tensor([1.0 if abs(x) > 1e-6 else 0.0 for x in transform])
        
        return targets, activity_labels, param_activity_labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        cordinates = torch.stack(data['feats_list'], dim=0)
        if self.inference:
            return (cordinates, data['targets'], data['vertices_list'], 
                   data['faces_list'], data['num_stages'], data['jaw_id'])
        return (cordinates, data['targets'], data['num_stages'], 
               data['activity_labels'], data['param_activity_labels'])