import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import logging
from dataset import JawTeethDataset
from models.DGCNN import DGCNN
from models.StageTransformer import StageTransformer
from models.OrthoDGCNN import OrthoDGCNNModel

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def setup_logging(log_file):
    logger = logging.getLogger('TrainLogger')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def train_model(args):
    logger = setup_logging(args.log_file)
    logger.info("Starting training with the following arguments:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
        logger.info(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    
    train_dataset = JawTeethDataset(
        args.data_dir, 
        max_stages=args.max_stages, 
        split='train', 
        train_ratio=args.train_ratio,
        inference=False,
        log_file=args.log_file
    )
    test_dataset = JawTeethDataset(
        args.data_dir, 
        max_stages=args.max_stages, 
        split='test', 
        train_ratio=args.train_ratio,
        inference=False,
        log_file=args.log_file
    )
    
    total_cases = len([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d)) and d.isdigit()])
    train_size = len(train_dataset)
    test_size = len(test_dataset)
    
    if train_size == 0 or test_size == 0:
        logger.error("Train or test dataset is empty. Check data_dir or train_ratio.")
        raise ValueError("Train or test dataset is empty. Check data_dir or train_ratio.")
    
    logger.info(f"Total cases in data_dir: {total_cases}")
    logger.info(f"Train set size: {train_size}")
    logger.info(f"Test set size: {test_size}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    dgcnn = DGCNN(in_channels=13, embed_dim=args.embed_dim, num_teeth=14, k=10).to(device)
    transformer = StageTransformer(
        d_model=14 * args.embed_dim,
        max_stages=args.max_stages,
        n_head=args.n_head,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers
    ).to(device)
    model = OrthoDGCNNModel(
        dgcnn, 
        transformer, 
        max_stages=args.max_stages,
        num_teeth=14,
        embed_dim=args.embed_dim,
        teacher_forcing=args.teacher_forcing
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "ortho_dgcnn.pth")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (cordinates, targets, true_num_stages) in enumerate(train_loader):
            cordinates, targets = cordinates.to(device), targets.to(device)
            true_num_stages = true_num_stages.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(cordinates, targets=targets)
                loss = 0
                for b in range(cordinates.shape[0]):
                    num_stages = true_num_stages[b].item()
                    loss += criterion(outputs[b, :num_stages], targets[b, :num_stages])
                loss = loss / cordinates.shape[0]
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.6f}")
        
        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}, Average Train Loss: {avg_train_loss:.6f}")
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_idx, (cordinates, targets, true_num_stages) in enumerate(test_loader):
                cordinates, targets = cordinates.to(device), targets.to(device)
                true_num_stages = true_num_stages.to(device)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(cordinates, targets=targets)
                    loss = 0
                    for b in range(cordinates.shape[0]):
                        num_stages = true_num_stages[b].item()
                        loss += criterion(outputs[b, :num_stages], targets[b, :num_stages])
                    loss = loss / cordinates.shape[0]
                
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        logger.info(f"Epoch {epoch+1}, Average Test Loss: {avg_test_loss:.6f}")
        
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved model checkpoint to {model_path}")
    
    logger.info("Training completed successfully.")
    logger.info(f"Final GPU memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OrthoDGCNNModel")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--max_stages', type=int, default=25)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--output_dir', type=str, default="output")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_head', type=int, default=16)
    parser.add_argument('--num_encoder_layers', type=int, default=6)
    parser.add_argument('--num_decoder_layers', type=int, default=6)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--log_file', type=str, default="train_log.txt")
    parser.add_argument('--teacher_forcing', action='store_true', default=False, help="Use teacher forcing during training")
    
    args = parser.parse_args()
    
    log_dir = os.path.dirname(args.log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    logger = setup_logging(args.log_file)
    logger.info("Initializing training script...")
    
    train_model(args)