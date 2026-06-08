import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss implementation to address high class imbalance
    (e.g., road background vs small pothole pixels).
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Tensor of shape (num_classes,)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs shape: (batch, num_classes, H, W)
        # targets shape: (batch, H, W)
        log_softmax = F.log_softmax(inputs, dim=1)
        ce_loss = F.nll_loss(log_softmax, targets, weight=self.alpha, reduction='none')
        
        # Calculate p_t
        all_p = torch.exp(log_softmax)
        # Gather the probability of the target classes
        # Reshape for gather
        targets_expanded = targets.unsqueeze(1) # (B, 1, H, W)
        p_t = torch.gather(all_p, 1, targets_expanded).squeeze(1) # (B, H, W)
        
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    """
    Dice Loss for multi-class semantic segmentation.
    """
    def __init__(self, smooth=1.0, ignore_index=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: (batch, num_classes, H, W)
        # targets: (batch, H, W)
        num_classes = inputs.size(1)
        inputs_softmax = F.softmax(inputs, dim=1)
        
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        total_loss = 0.0
        classes_counted = 0
        
        for class_idx in range(num_classes):
            if self.ignore_index is not None and class_idx == self.ignore_index:
                continue
                
            pred = inputs_softmax[:, class_idx, :, :]
            target = targets_one_hot[:, class_idx, :, :]
            
            intersection = (pred * target).sum(dim=(1, 2))
            union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
            
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            class_loss = 1.0 - dice
            
            total_loss += class_loss.mean()
            classes_counted += 1
            
        return total_loss / classes_counted if classes_counted > 0 else torch.tensor(0.0, device=inputs.device)


class RoadHazardLoss(nn.Module):
    """
    Combined Segmentation Loss: 60% Focal Loss + 40% Dice Loss.
    """
    def __init__(self, alpha=None, gamma=2.0, weight_focal=0.6, weight_dice=0.4):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice

    def forward(self, inputs, targets):
        focal_loss = self.focal(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.weight_focal * focal_loss + self.weight_dice * dice_loss


def get_segformer_b3_model(pretrained=True):
    """
    Creates and returns a SegFormer-B3 model.
    If pretrained is True, it downloads pre-trained mit-b3 weights.
    """
    # Use HuggingFace's SegformerForSemanticSegmentation
    # Since we are using standard mit-b3 as a backbone, we'll initialize the config and set the classification head
    if pretrained:
        print(f"Loading pretrained SegFormer-B3 backbone: {config.SEGFORMER_BACKBONE}")
        # Note: SegformerForSemanticSegmentation from_pretrained will load the model weights,
        # and ignore the classification head since we are setting num_labels dynamically.
        model = SegformerForSemanticSegmentation.from_pretrained(
            config.SEGFORMER_BACKBONE,
            num_labels=config.NUM_CLASSES,
            ignore_mismatched_sizes=True
        )
    else:
        print("Initializing new SegFormer-B3 configuration...")
        configuration = SegformerConfig.from_pretrained(
            config.SEGFORMER_BACKBONE,
            num_labels=config.NUM_CLASSES
        )
        model = SegformerForSemanticSegmentation(configuration)
        
    return model
