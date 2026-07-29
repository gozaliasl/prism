#!/usr/bin/env python3
"""
ML-Based Arc and Source Detection for Lensed Images

This module provides machine learning models for detecting lensed arcs and sources
in simulated JWST images. Can be trained on labeled data and used for:
1. Arc/source detection
2. Photometry extraction
3. Image annotation

Supports:
- CNN-based arc detection
- Transfer learning from existing models
- Self-supervised learning on simulated data
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import ML libraries
TORCH_AVAILABLE = False
nn = None
torch = None
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not available - ML detection disabled")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Try to import photutils for photometry
try:
    from photutils.segmentation import detect_sources, SourceCatalog, deblend_sources
    from photutils.background import Background2D, MedianBackground
    from photutils.aperture import CircularAperture, aperture_photometry
    PHOTUTILS_AVAILABLE = True
except ImportError:
    try:
        from photutils import detect_sources, deblend_sources
        from photutils.segmentation import SourceCatalog
        from photutils.background import Background2D, MedianBackground
        from photutils.aperture import CircularAperture, aperture_photometry
        PHOTUTILS_AVAILABLE = True
    except ImportError:
        PHOTUTILS_AVAILABLE = False
        print("[WARNING] photutils not available - photometry disabled")


if TORCH_AVAILABLE:
    class ArcDetectionCNN(nn.Module):
        """
        CNN for detecting lensed arcs in images.
        
        Architecture:
        - Input: 64x64 image patches
        - Convolutional layers for feature extraction
        - Classification: arc vs non-arc
        - Regression: arc properties (position, elongation, curvature)
        """
        def __init__(self, num_classes=2):
            super(ArcDetectionCNN, self).__init__()
            
            # Feature extraction
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            
            # Classification head
            self.fc1 = nn.Linear(128 * 8 * 8, 256)
            self.fc2 = nn.Linear(256, num_classes)
            
            # Regression head (for arc properties)
            self.reg_fc1 = nn.Linear(128 * 8 * 8, 128)
            self.reg_fc2 = nn.Linear(128, 4)  # x, y, elongation, curvature
            
        def forward(self, x):
            # Feature extraction
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.pool(torch.relu(self.conv3(x)))
            
            x = x.view(-1, 128 * 8 * 8)
            
            # Classification
            cls = torch.relu(self.fc1(x))
            cls = self.fc2(cls)
            
            # Regression
            reg = torch.relu(self.reg_fc1(x))
            reg = self.reg_fc2(reg)
            
            return cls, reg
else:
    # Dummy class if PyTorch not available
    class ArcDetectionCNN:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch not available - cannot create ArcDetectionCNN")


def extract_photometry(
    image: np.ndarray,
    positions: List[Tuple[float, float]],
    aperture_radius: float = 3.0,
    background_subtract: bool = True
) -> List[Dict]:
    """
    Extract photometry for detected sources using circular apertures.
    
    Args:
        image: 2D image array
        positions: List of (x, y) positions in pixels
        aperture_radius: Aperture radius in pixels
        background_subtract: Whether to subtract background
    
    Returns:
        List of photometry dictionaries
    """
    if not PHOTUTILS_AVAILABLE:
        return []
    
    photometry_results = []
    
    # Estimate background if needed
    background = None
    if background_subtract:
        try:
            bkg_estimator = MedianBackground()
            bkg = Background2D(image, (50, 50), filter_size=(3, 3),
                             bkg_estimator=bkg_estimator)
            background = bkg.background
        except:
            background = np.median(image)  # Simple median background
    
    for x, y in positions:
        try:
            # Create circular aperture
            aperture = CircularAperture((x, y), r=aperture_radius)
            
            # Perform photometry
            phot_table = aperture_photometry(image, aperture)
            flux = phot_table['aperture_sum'][0]
            
            # Subtract background if requested
            if background_subtract and background is not None:
                if isinstance(background, np.ndarray):
                    # Local background from aperture area
                    from scipy import ndimage
                    mask = np.zeros_like(image, dtype=bool)
                    y_int, x_int = int(y), int(x)
                    y_int = np.clip(y_int, 0, image.shape[0] - 1)
                    x_int = np.clip(x_int, 0, image.shape[1] - 1)
                    local_bkg = background[y_int, x_int] * np.pi * aperture_radius**2
                else:
                    local_bkg = background * np.pi * aperture_radius**2
                flux -= local_bkg
            
            # Convert to magnitude (assuming zero point)
            # magnitude = -2.5 * log10(flux) + zeropoint
            # For now, just return flux
            
            photometry_results.append({
                'x': x,
                'y': y,
                'flux': float(flux),
                'aperture_radius': aperture_radius,
                'magnitude': None  # Can be calculated with zeropoint
            })
        except Exception as e:
            print(f"[WARNING] Photometry failed for position ({x}, {y}): {e}")
            continue
    
    return photometry_results


def detect_with_ml_model(
    image: np.ndarray,
    model_path: Optional[Path] = None,
    patch_size: int = 64,
    stride: int = 16,
    confidence_threshold: float = 0.7
) -> List[Dict]:
    """
    Detect arcs/sources using a trained ML model.
    
    Args:
        image: 2D image array
        model_path: Path to trained model (if None, uses default or falls back)
        patch_size: Size of image patches to classify
        stride: Stride for sliding window
        confidence_threshold: Minimum confidence for detection
    
    Returns:
        List of detected arc/source candidates
    """
    if not TORCH_AVAILABLE:
        print("[WARNING] PyTorch not available - ML detection disabled")
        return []
    
    # Load model if available
    model = None
    if model_path and model_path.exists():
        try:
            model = ArcDetectionCNN()
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
        except Exception as e:
            print(f"[WARNING] Could not load model: {e}")
            model = None
    
    if model is None:
        print("[INFO] No trained model available - using traditional detection")
        return []
    
    # Sliding window detection
    candidates = []
    h, w = image.shape
    
    # Normalize image
    image_norm = (image - image.min()) / (image.max() - image.min() + 1e-10)
    
    with torch.no_grad():
        for y in range(0, h - patch_size, stride):
            for x in range(0, w - patch_size, stride):
                # Extract patch
                patch = image_norm[y:y+patch_size, x:x+patch_size]
                
                # Skip if patch is too dark
                if patch.max() < 0.1:
                    continue
                
                # Convert to tensor
                patch_tensor = torch.FloatTensor(patch).unsqueeze(0).unsqueeze(0)
                
                # Predict
                cls_logits, reg_output = model(patch_tensor)
                cls_probs = torch.softmax(cls_logits, dim=1)
                confidence = cls_probs[0, 1].item()  # Probability of being an arc
                
                if confidence > confidence_threshold:
                    # Get predicted properties
                    pred_x = x + patch_size // 2 + reg_output[0, 0].item()
                    pred_y = y + patch_size // 2 + reg_output[0, 1].item()
                    elongation = reg_output[0, 2].item()
                    curvature = reg_output[0, 3].item()
                    
                    candidates.append({
                        'x': pred_x,
                        'y': pred_y,
                        'confidence': confidence,
                        'elongation': elongation,
                        'curvature': curvature,
                        'method': 'ml_cnn'
                    })
    
    return candidates


def prepare_training_data(
    image_dir: Path,
    annotation_file: Path,
    patch_size: int = 64,
    output_dir: Path = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare training data from images and annotations.
    
    Args:
        image_dir: Directory containing images
        annotation_file: CSV file with annotations (x, y, is_arc)
        patch_size: Size of patches to extract
        output_dir: Directory to save prepared data
    
    Returns:
        Tuple of (patches, labels)
    """
    import pandas as pd
    from PIL import Image
    
    # Load annotations
    annotations = pd.read_csv(annotation_file)
    
    patches = []
    labels = []
    
    for idx, row in annotations.iterrows():
        image_path = image_dir / row['image_file']
        if not image_path.exists():
            continue
        
        # Load image
        img = Image.open(image_path)
        if img.mode == 'RGB':
            img_array = np.array(img.convert('L')) / 255.0
        else:
            img_array = np.array(img) / 255.0
        
        x, y = int(row['x']), int(row['y'])
        is_arc = int(row.get('is_arc', 0))
        
        # Extract patch around position
        half_size = patch_size // 2
        y_min = max(0, y - half_size)
        y_max = min(img_array.shape[0], y + half_size)
        x_min = max(0, x - half_size)
        x_max = min(img_array.shape[1], x + half_size)
        
        patch = img_array[y_min:y_max, x_min:x_max]
        
        # Pad if necessary
        if patch.shape != (patch_size, patch_size):
            padded = np.zeros((patch_size, patch_size))
            pad_y = (patch_size - patch.shape[0]) // 2
            pad_x = (patch_size - patch.shape[1]) // 2
            padded[pad_y:pad_y+patch.shape[0], pad_x:pad_x+patch.shape[1]] = patch
            patch = padded
        
        patches.append(patch)
        labels.append(is_arc)
    
    return np.array(patches), np.array(labels)


def train_arc_detector(
    training_data_dir: Path,
    model_output_path: Path,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001
):
    """
    Train CNN model for arc detection.
    
    Args:
        training_data_dir: Directory with training images and annotations
        model_output_path: Where to save trained model
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    """
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available - cannot train model")
        return
    
    # Prepare data
    annotation_file = training_data_dir / 'annotations.csv'
    image_dir = training_data_dir / 'images'
    
    if not annotation_file.exists():
        print(f"❌ Annotation file not found: {annotation_file}")
        return
    
    print("📊 Preparing training data...")
    patches, labels = prepare_training_data(image_dir, annotation_file)
    
    if len(patches) == 0:
        print("❌ No training data found")
        return
    
    print(f"✅ Prepared {len(patches)} training samples")
    
    # Convert to tensors
    X = torch.FloatTensor(patches).unsqueeze(1)  # Add channel dimension
    y = torch.LongTensor(labels)
    
    # Split train/val
    split_idx = int(0.8 * len(patches))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Create model
    model = ArcDetectionCNN(num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("🎯 Training model...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs, _ = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Validation
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs, _ = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                val_pred = torch.argmax(val_outputs, dim=1)
                val_acc = (val_pred == y_val).float().mean()
                print(f"Epoch {epoch+1}/{epochs}: Loss={loss.item():.4f}, Val Loss={val_loss.item():.4f}, Val Acc={val_acc.item():.4f}")
    
    # Save model
    torch.save(model.state_dict(), model_output_path)
    print(f"✅ Model saved to: {model_output_path}")


def detect_lensed_images_ml_enhanced(
    image: np.ndarray,
    lens_center: Tuple[float, float],
    einstein_radius_pix: float,
    model_path: Optional[Path] = None,
    use_ml: bool = True,
    use_traditional: bool = True,
    pixel_scale: float = 0.03,
    numpix: int = 300
) -> List[Dict]:
    """
    Enhanced detection combining ML and traditional methods.
    
    Args:
        image: 2D image array
        lens_center: (x, y) lens center position
        einstein_radius_pix: Einstein radius in pixels
        model_path: Path to trained ML model
        use_ml: Use ML detection
        use_traditional: Use traditional detection
        pixel_scale: Pixel scale
        numpix: Image size
    
    Returns:
        List of detected candidates with photometry
    """
    candidates = []
    
    # ML-based detection
    if use_ml and TORCH_AVAILABLE:
        ml_candidates = detect_with_ml_model(image, model_path)
        candidates.extend(ml_candidates)
    
    # Traditional detection (as fallback or complement)
    if use_traditional:
        from prism.lensing.detect_lensed_images import identify_lensed_images_hybrid
        traditional_candidates = identify_lensed_images_hybrid(
            image, lens_center, einstein_radius_pix,
            pixel_scale=pixel_scale, numpix=numpix
        )
        candidates.extend(traditional_candidates)
    
    # Extract photometry for all candidates
    positions = [(c['x'], c['y']) for c in candidates]
    photometry = extract_photometry(image, positions)
    
    # Merge photometry with candidates
    for i, cand in enumerate(candidates):
        if i < len(photometry):
            cand.update(photometry[i])
    
    return candidates

