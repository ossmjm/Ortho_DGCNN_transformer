import torch
from torch.utils.data import DataLoader
import os
import argparse
import logging
import pandas as pd
from dataset import CumulativeJawTeethDataset
from models.OrthoDGCNN import OrthoDGCNNModel

def setup_logging(log_file):
    logger = logging.getLogger('InferenceLogger')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def custom_collate_fn(batch):
    logger = logging.getLogger('InferenceLogger')
    
    jaw_ids = [item[0] for item in batch]
    feats = [item[1] for item in batch]
    cumulative_transforms = [item[2] for item in batch]
    
    try:
        feats = torch.stack(feats)
        cumulative_transforms = torch.stack(cumulative_transforms)
        
        logger.debug(f"Collated batch: jaw_ids={len(jaw_ids)}, feats_shape={feats.shape}")
        
        return jaw_ids, feats, cumulative_transforms
    except Exception as e:
        logger.error(f"Error in custom_collate_fn: {str(e)}")
        raise

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with OrthoDGCNN Cumulative Model")
    parser.add_argument('--data_dir', type=str, default='./Data', help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for inference')
    parser.add_argument('--num_teeth', type=int, default=14, help='Number of teeth')
    parser.add_argument('--num_points', type=int, default=256, help='Number of points per tooth')
    parser.add_argument('--channels', type=int, default=13, help='Number of feature channels')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train/validation split ratio')
    parser.add_argument('--log_file', type=str, default='inference_log.txt', help='Log file path')
    parser.add_argument('--model_path', type=str, default='./output/best_model.pth', help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory for Excel')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='Path to cache directory')
    parser.add_argument('--embed_dim', type=int, default=384, help='Embedding dimension')
    parser.add_argument('--k', type=int, default=20, help='Number of k in DGCNN')
    return parser.parse_args()

def infer_model(args):
    logger = setup_logging(args.log_file)
    logger.info("Inference with the following arguments:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    torch.cuda.empty_cache()
    logger.info(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    
    FDI_numbers = [31, 32, 33, 34, 35, 36, 37, 41, 42, 43, 44, 45, 46, 47]
    
    dataset = CumulativeJawTeethDataset(
        data_dir=args.data_dir,
        num_teeth=args.num_teeth,
        num_points=args.num_points,
        channels=args.channels,
        split='test',
        train_ratio=args.train_ratio,
        inference=True,
        cache_dir=args.cache_dir,
        log_file=args.log_file
    )
    logger.info(f"Dataset size: {len(dataset)}")
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    logger.info(f"Number of batches: {len(data_loader)}")
    
    model = OrthoDGCNNModel(
        num_teeth=args.num_teeth,
        num_points=args.num_points,
        channels=args.channels,
        embed_dim=args.embed_dim,
        k=args.k
    ).to(device)
    
    logger.info(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    os.makedirs(args.output_dir, exist_ok=True)
    cumulative_transform_data = []
    
    with torch.no_grad():
        for batch_idx, (jaw_ids, feats, _) in enumerate(data_loader):
            feats = feats.to(device)
            batch_size = feats.size(0)
            
            if torch.isnan(feats).any() or torch.isinf(feats).any():
                logger.warning(f"Batch {batch_idx+1}: feats contains nan/inf")
                continue
            
            with torch.amp.autocast('cuda'):
                cumulative_transforms, _, _ = model(feats)
            
            logger.debug(f"Batch {batch_idx+1}: cumulative_transforms_shape={cumulative_transforms.shape}")
            
            if torch.isnan(cumulative_transforms).any() or torch.isinf(cumulative_transforms).any():
                logger.warning(f"Batch {batch_idx+1}: cumulative_transforms contains nan/inf")
                continue
            
            for b in range(batch_size):
                jaw_id = jaw_ids[b]
                for tooth_idx, fdi in enumerate(FDI_numbers):
                    cumulative_translation = cumulative_transforms[b, tooth_idx, :3]
                    cumulative_rotation = cumulative_transforms[b, tooth_idx, 3:]
                    cumulative_transform_data.append({
                        'Jaw_ID': jaw_id,
                        'Tooth_ID': fdi,
                        'Left/Right (mm)': cumulative_translation[0].item(),
                        'Forward/Backward (mm)': cumulative_translation[1].item(),
                        'Extrude/Intrude (mm)': cumulative_translation[2].item(),
                        'Buccal/Lingual (degrees)': cumulative_rotation[0].item(),
                        'Mesial/Distal (degrees)': cumulative_rotation[1].item(),
                        'Rotation (degrees)': cumulative_rotation[2].item()
                    })
    
    cumulative_df = pd.DataFrame(cumulative_transform_data)
    cumulative_excel_path = os.path.join(args.output_dir, 'predicted_cumulative_transformations.xlsx')
    try:
        cumulative_df.to_excel(cumulative_excel_path, index=False)
        logger.info(f"Saved cumulative transformation data to {cumulative_excel_path}")
    except Exception as e:
        logger.error(f"Error saving cumulative Excel file: {e}")
    
    logger.info("Inference completed")

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    infer_model(args)