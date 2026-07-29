#!/usr/bin/env python3
"""
U-Net Segmentation Model for Lensed Source Detection

This module implements a U-Net architecture for pixel-level detection of lensed sources.
Trained on simulation data where we know the exact positions, it learns to identify
lensed source pixels in images.

Similar to YOLO/object detection but for pixel-level segmentation.
"""

import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Try to import PyTorch
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not available - segmentation model disabled")


# Define UNet class (only works if PyTorch is available)
UNet = None
if TORCH_AVAILABLE:
    class UNet(nn.Module):
        """
        U-Net architecture for lensed source segmentation.
        
        Input: Image patch (H x W)
        Output: Segmentation mask (H x W) where 1 = lensed source, 0 = background
        """
        def __init__(self, in_channels=1, out_channels=1):
            super(UNet, self).__init__()
            
            # Encoder (downsampling path)
            self.enc1 = self._conv_block(in_channels, 64)
            self.enc2 = self._conv_block(64, 128)
            self.enc3 = self._conv_block(128, 256)
            self.enc4 = self._conv_block(256, 512)
            
            # Bottleneck
            self.bottleneck = self._conv_block(512, 1024)
            
            # Decoder (upsampling path)
            self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
            self.dec4 = self._conv_block(1024, 512)
            self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
            self.dec3 = self._conv_block(512, 256)
            self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.dec2 = self._conv_block(256, 128)
            self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dec1 = self._conv_block(128, 64)
            
            # Final output layer
            self.final = nn.Conv2d(64, out_channels, 1)
            self.sigmoid = nn.Sigmoid()
            
        def _conv_block(self, in_channels, out_channels):
            """Convolutional block: Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU"""
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        def forward(self, x):
            # Encoder
            enc1 = self.enc1(x)
            enc2 = self.enc2(F.max_pool2d(enc1, 2))
            enc3 = self.enc3(F.max_pool2d(enc2, 2))
            enc4 = self.enc4(F.max_pool2d(enc3, 2))
            
            # Bottleneck
            bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
            
            # Decoder with skip connections
            dec4 = self.up4(bottleneck)
            dec4 = torch.cat([dec4, enc4], dim=1)
            dec4 = self.dec4(dec4)
            
            dec3 = self.up3(dec4)
            dec3 = torch.cat([dec3, enc3], dim=1)
            dec3 = self.dec3(dec3)
            
            dec2 = self.up2(dec3)
            dec2 = torch.cat([dec2, enc2], dim=1)
            dec2 = self.dec2(dec2)
            
            dec1 = self.up1(dec2)
            dec1 = torch.cat([dec1, enc1], dim=1)
            dec1 = self.dec1(dec1)
            
            # Final output
            output = self.final(dec1)
            output = self.sigmoid(output)
            
            return output


def train_segmentation_model(
    training_dir: Path,
    model_output_path: Path,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    device: str = 'cpu'
):
    """
    Train U-Net model for lensed source segmentation.
    
    Args:
        training_dir: Directory with images/ and masks/ subdirectories
        model_output_path: Where to save trained model
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: 'cpu' or 'cuda'
    """
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available - cannot train model")
        return
    
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    
    # Custom dataset
    class SegmentationDataset(Dataset):
        def __init__(self, images_dir, masks_dir, metadata_df):
            self.images_dir = images_dir
            self.masks_dir = masks_dir
            self.metadata = metadata_df
        
        def __len__(self):
            return len(self.metadata)
        
        def __getitem__(self, idx):
            row = self.metadata.iloc[idx]
            img = np.load(self.images_dir / row['image_file'].split('/')[-1])
            mask = np.load(self.masks_dir / row['mask_file'].split('/')[-1])
            
            # Convert to tensors
            img_tensor = torch.FloatTensor(img).unsqueeze(0)  # Add channel dimension
            mask_tensor = torch.FloatTensor(mask).unsqueeze(0)
            
            return img_tensor, mask_tensor
    
    # Load metadata
    if not PANDAS_AVAILABLE:
        print("❌ Error: pandas not available")
        return
    
    metadata_file = training_dir / 'training_metadata.csv'
    if not metadata_file.exists():
        print(f"❌ Error: Training metadata not found: {metadata_file}")
        return
    
    metadata = pd.read_csv(metadata_file)
    
    # Split train/val
    split_idx = int(0.8 * len(metadata))
    train_metadata = metadata[:split_idx]
    val_metadata = metadata[split_idx:]
    
    # Create datasets
    train_dataset = SegmentationDataset(
        training_dir / 'images',
        training_dir / 'masks',
        train_metadata
    )
    val_dataset = SegmentationDataset(
        training_dir / 'images',
        training_dir / 'masks',
        val_metadata
    )
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = UNet(in_channels=1, out_channels=1)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"🎯 Training U-Net segmentation model...")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Epochs: {epochs}, Batch size: {batch_size}")
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_output_path)
    
    print(f"✅ Model saved to: {model_output_path}")
    print(f"   Best validation loss: {best_val_loss:.4f}")


def detect_lensed_sources_with_model(
    image: np.ndarray,
    model_path: Path,
    patch_size: int = 128,
    stride: int = 64,
    threshold: float = 0.5,
    device: str = 'cpu'
) -> Tuple[np.ndarray, list]:
    """
    Detect lensed sources using trained U-Net model.
    
    Args:
        image: 2D image array
        model_path: Path to trained model
        patch_size: Size of patches to process
        stride: Stride for sliding window
        threshold: Probability threshold for detection
        device: 'cpu' or 'cuda'
    
    Returns:
        - Full segmentation mask (2D array, 1 = lensed source)
        - List of detected source regions (dicts with positions, areas, etc.)
    """
    if not TORCH_AVAILABLE:
        print("[WARNING] PyTorch not available")
        return np.zeros_like(image), []
    
    # Load model
    model = UNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model = model.to(device)
    
    h, w = image.shape
    full_mask = np.zeros((h, w), dtype=np.float32)
    count_mask = np.zeros((h, w), dtype=np.int32)
    
    # Sliding window inference
    with torch.no_grad():
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                # Extract patch
                patch = image[y:y+patch_size, x:x+patch_size]
                
                # Normalize
                patch_norm = (patch - patch.min()) / (patch.max() - patch.min() + 1e-10)
                
                # Convert to tensor
                patch_tensor = torch.FloatTensor(patch_norm).unsqueeze(0).unsqueeze(0).to(device)
                
                # Predict
                output = model(patch_tensor)
                pred_mask = output.cpu().numpy()[0, 0]
                
                # Add to full mask (average overlapping regions)
                full_mask[y:y+patch_size, x:x+patch_size] += pred_mask
                count_mask[y:y+patch_size, x:x+patch_size] += 1
    
    # Average overlapping regions
    full_mask = full_mask / (count_mask + 1e-10)
    
    # Threshold
    binary_mask = (full_mask > threshold).astype(np.uint8)
    
    # Extract connected components (individual sources)
    try:
        from skimage import measure
        SKIMAGE_AVAILABLE = True
    except ImportError:
        SKIMAGE_AVAILABLE = False
    
    if SKIMAGE_AVAILABLE:
        labeled_mask = measure.label(binary_mask)
        regions = measure.regionprops(labeled_mask)
        
        detected_sources = []
        for region in regions:
            if region.area < 10:  # Too small
                continue
            
            # Get centroid
            y_centroid, x_centroid = region.centroid
            
            detected_sources.append({
                'x': float(x_centroid),
                'y': float(y_centroid),
                'area': int(region.area),
                'confidence': float(full_mask[int(y_centroid), int(x_centroid)]),
                'bbox': region.bbox  # (min_row, min_col, max_row, max_col)
            })
        
        return binary_mask, detected_sources
    else:
        return binary_mask, []

