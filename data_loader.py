# data_loader.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from config import *

class OHLCVDataset(Dataset):
    """
    Dataset from OHLCV CSV files.
    Expects columns: timestamp, open, high, low, close, volume (optional).
    """
    def __init__(self, csv_path, seq_len=SEQ_LEN, train=True, split_ratio=0.8):
        df = pd.read_csv(csv_path)
        # Assume OHLCV columns: open, high, low, close, volume
        self.data = df[['open','high','low','close']].values.astype(np.float32)
        # Simple normalisation (percentage change)
        self.data = (self.data / self.data[0]) - 1.0   # relative to first candle
        
        # Create sequences
        self.seq_len = seq_len
        self.samples = []
        for i in range(len(self.data)-seq_len):
            x = self.data[i:i+seq_len]
            # Label: next candle direction (simplified)
            future_close = self.data[i+seq_len, 3]
            current_close = self.data[i+seq_len-1, 3]
            if future_close > current_close * 1.001:
                label = 0   # LONG
            elif future_close < current_close * 0.999:
                label = 1   # SHORT
            else:
                label = 2   # NO TRADE
            self.samples.append((x, label))
        
        split = int(len(self.samples)*split_ratio)
        if train:
            self.samples = self.samples[:split]
        else:
            self.samples = self.samples[split:]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y, dtype=torch.long)

def create_dataloader(csv_path, batch_size=BATCH_SIZE, train=True):
    dataset = OHLCVDataset(csv_path, train=train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)

def load_image(image_path):
    """Load and preprocess chart image for ViT."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img / 255.0
    img = torch.tensor(img).permute(2, 0, 1).float()
    return img