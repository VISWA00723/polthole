import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import albumentations as A

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from segmentation.dataset import RoadHazardDataset, collect_dataset_paths, oversample_potholes
from segmentation.model import get_segformer_model, RoadHazardLoss, MultiTaskRoadHazardModel

def compute_iou(pred, target, num_classes):
    """Computes IoU for each class."""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        
        if union == 0:
            ious.append(float('nan'))  # Class not present in either pred or target
        else:
            ious.append(intersection / union)
            
    return ious

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    
    loop = tqdm(loader, desc="Training")
    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device)
        
        B = images.shape[0]
        severity_targets = []
        road_score_targets = []
        risk_score_targets = []
        
        # Compute multi-task targets on the fly from ground truth masks
        for i in range(B):
            mask_i = masks[i]
            num_hazard_pixels = (mask_i == 1).sum().item() + (mask_i == 3).sum().item() + (mask_i == 4).sum().item()
            
            # Severity target
            if num_hazard_pixels > 2000:
                sev = 2  # High
            elif num_hazard_pixels > 200:
                sev = 1  # Medium
            else:
                sev = 0  # Low
            severity_targets.append(sev)
            
            # Road score target
            pothole_ratio = (mask_i == 1).sum().item() / mask_i.numel()
            crack_ratio = (mask_i == 2).sum().item() / mask_i.numel()
            water_pothole_ratio = (mask_i == 3).sum().item() / mask_i.numel()
            manhole_ratio = (mask_i == 4).sum().item() / mask_i.numel()
            
            deduction = (
                4.5 * pothole_ratio +
                1.5 * crack_ratio +
                5.0 * water_pothole_ratio +
                1.0 * manhole_ratio
            )
            score = max(0.0, 10.0 - deduction * 100.0)
            road_score_targets.append(score)
            
            # Risk score target
            risk = min(100.0, num_hazard_pixels / 50.0)
            risk_score_targets.append(risk)
            
        severity_targets = torch.tensor(severity_targets, dtype=torch.long, device=device)
        road_score_targets = torch.tensor(road_score_targets, dtype=torch.float32, device=device).unsqueeze(1)
        risk_score_targets = torch.tensor(risk_score_targets, dtype=torch.float32, device=device).unsqueeze(1)
        
        # Standard physical inputs for risk head during training: [area, depth, distance, speed]
        physical_inputs = torch.zeros(B, 4, device=device)
        physical_inputs[:, 3] = 15.0  # speed km/h
        
        optimizer.zero_grad()
        
        logits, severity_logits, road_score, risk_score = model(images, physical_inputs)
        
        logits_upscaled = torch.nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        
        # Compute multi-task losses
        loss_seg = criterion(logits_upscaled, masks)
        loss_sev = F.cross_entropy(severity_logits, severity_targets)
        loss_road = F.mse_loss(road_score, road_score_targets)
        loss_risk = F.mse_loss(risk_score, risk_score_targets)
        
        # Combined loss
        loss = loss_seg + 0.1 * loss_sev + 0.1 * loss_road + 0.01 * loss_risk
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, device, num_classes):
    model.eval()
    all_ious = []
    
    loop = tqdm(loader, desc="Validation")
    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device)
        
        logits, _, _, _ = model(images)
        
        logits_upscaled = torch.nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        
        preds = torch.argmax(logits_upscaled, dim=1)
        
        batch_ious = compute_iou(preds, masks, num_classes)
        all_ious.append(batch_ious)
        
    all_ious = np.array(all_ious)
    mean_class_ious = np.nanmean(all_ious, axis=0)
    mIoU = np.nanmean(mean_class_ious)
    
    return mIoU, mean_class_ious

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Collect dataset files
    print(f"Scanning dataset directory: {config.DATASET_DIR}...")
    dataset_files = collect_dataset_paths(config.DATASET_DIR)
    print(f"Found {len(dataset_files)} total images.")
    
    if len(dataset_files) == 0:
        print("WARNING: No images found. Training cannot proceed.")
        sys.exit(1)
        
    # 2. Split into train and validation sets (80% / 20%)
    train_size = int(0.8 * len(dataset_files))
    val_size = len(dataset_files) - train_size
    train_paths, val_paths = random_split(dataset_files, [train_size, val_size])
    
    # Get actual list items from split subsets
    train_paths_list = [dataset_files[i] for i in train_paths.indices]
    val_paths_list = [dataset_files[i] for i in val_paths.indices]
    
    # Apply oversampling on training paths list to counter class imbalance
    print(f"Oversampling training set (original size: {len(train_paths_list)})...")
    train_paths_list = oversample_potholes(train_paths_list, oversample_factor=3)
    print(f"Oversampled training set size: {len(train_paths_list)}")
    
    # 3. Define Albumentations augmentations
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.MotionBlur(p=0.2),
        A.GaussNoise(p=0.2),
        A.RandomRain(p=0.1),
        A.RandomFog(p=0.1),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
    ])
    
    # Create datasets and Dataloaders
    train_dataset = RoadHazardDataset(train_paths_list, transform=train_transform, img_size=(512, 512))
    val_dataset = RoadHazardDataset(val_paths_list, img_size=(512, 512))
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    # 4. Instantiate base SegFormer and wrap in Multi-Task model
    base_model = get_segformer_model(pretrained=True)
    model = MultiTaskRoadHazardModel(base_model, hidden_dim=512)
    model.to(device)
    
    # 5. Define Loss & Optimizer
    # Class weights to handle background vs small hazard instances
    alpha_weights = torch.tensor([0.1, 1.5, 1.0, 2.0, 1.5], device=device)
    criterion = RoadHazardLoss(alpha=alpha_weights, mode='ce')
    
    optimizer = optim.AdamW(model.parameters(), lr=6e-5, weight_decay=1e-2)
    
    # Schedulers
    epochs = 50
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scheduler_plateau = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Early stopping configuration
    best_mIoU = -1.0
    patience = 10
    epochs_no_improve = 0
    
    print("Starting Training Process...")
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler_cosine.step()
        
        print(f"Epoch {epoch} complete. Train Loss: {train_loss:.4f}")
        
        mIoU, class_ious = validate(model, val_loader, device, config.NUM_CLASSES)
        print(f"Validation mIoU: {mIoU:.4f}")
        for class_id, class_name in config.CLASS_MAP.items():
            print(f"  - Class {class_name} IoU: {class_ious[class_id]:.4f}")
            
        # Update plateau scheduler
        scheduler_plateau.step(mIoU)
        
        # Save best model
        if mIoU > best_mIoU:
            best_mIoU = mIoU
            epochs_no_improve = 0
            torch.save(model.state_dict(), config.SEGMENTATION_WEIGHTS)
            print(f"Saved new best model checkpoint to {config.SEGMENTATION_WEIGHTS} with mIoU {mIoU:.4f}")
        else:
            epochs_no_improve += 1
            print(f"Validation mIoU did not improve. Early stopping patience: {patience - epochs_no_improve}/{patience}")
            
        # Check early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch} epochs of no improvement.")
            break

    print(f"\nTraining completed. Best validation mIoU: {best_mIoU:.4f}")

if __name__ == "__main__":
    main()
