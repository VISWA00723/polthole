import os
import sys
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from segmentation.dataset import RoadHazardDataset, collect_dataset_paths
from segmentation.model import get_segformer_b3_model, RoadHazardLoss

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
        
        optimizer.zero_grad()
        
        outputs = model(pixel_values=images)
        logits = outputs.logits  # Segformer outputs logits
        
        # Segformer outputs are typically 1/4 of input size, so we upscale logits
        logits_upscaled = torch.nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        
        loss = criterion(logits_upscaled, masks)
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
        
        outputs = model(pixel_values=images)
        logits = outputs.logits
        
        logits_upscaled = torch.nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        
        preds = torch.argmax(logits_upscaled, dim=1)
        
        # Calculate IoU for this batch
        batch_ious = compute_iou(preds, masks, num_classes)
        all_ious.append(batch_ious)
        
    # Average class IoUs across validation batches
    all_ious = np.array(all_ious)
    mean_class_ious = np.nanmean(all_ious, axis=0)
    mIoU = np.nanmean(mean_class_ious)
    
    return mIoU, mean_class_ious

def main():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Collect dataset files
    print(f"Scanning dataset directory: {config.DATASET_DIR}...")
    dataset_files = collect_dataset_paths(config.DATASET_DIR)
    print(f"Found {len(dataset_files)} total images for training.")
    
    if len(dataset_files) == 0:
        print("WARNING: No images found. Training cannot proceed.")
        print("Please run 'python src/utils/download_datasets.py' or add datasets manually.")
        sys.exit(1)
        
    # 2. Split into train and validation sets (80% / 20%)
    train_size = int(0.8 * len(dataset_files))
    val_size = len(dataset_files) - train_size
    train_paths, val_paths = random_split(dataset_files, [train_size, val_size])
    
    # 3. Create datasets and Dataloaders
    train_dataset = RoadHazardDataset(train_paths, img_size=(512, 512))
    val_dataset = RoadHazardDataset(val_paths, img_size=(512, 512))
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    # 4. Instantiate Model
    model = get_segformer_b3_model(pretrained=True)
    model.to(device)
    
    # 5. Define Loss & Optimizer
    # Class weights to handle background vs small hazard instances
    # Background (0), Pothole (1), Crack (2), Water Pothole (3), Manhole (4)
    alpha_weights = torch.tensor([0.1, 1.0, 1.0, 1.2, 0.8], device=device)
    criterion = RoadHazardLoss(alpha=alpha_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=6e-5, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    
    # 6. Training Loop
    epochs = 10
    best_mIoU = -1.0
    
    print("Starting Training Process...")
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch} complete. Train Loss: {train_loss:.4f}")
        
        mIoU, class_ious = validate(model, val_loader, device, config.NUM_CLASSES)
        print(f"Validation mIoU: {mIoU:.4f}")
        for class_id, class_name in config.CLASS_MAP.items():
            print(f"  - Class {class_name} IoU: {class_ious[class_id]:.4f}")
            
        # Save best model
        if mIoU > best_mIoU:
            best_mIoU = mIoU
            torch.save(model.state_dict(), config.SEGMENTATION_WEIGHTS)
            print(f"Saved new best model checkpoint to {config.SEGMENTATION_WEIGHTS} with mIoU {mIoU:.4f}")

    print(f"\nTraining completed. Best validation mIoU: {best_mIoU:.4f}")

if __name__ == "__main__":
    main()
