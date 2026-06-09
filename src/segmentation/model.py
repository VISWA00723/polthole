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
    Combined Segmentation Loss: Supports both Weighted CrossEntropy + Dice Loss
    and Focal Loss + Dice Loss.
    """
    def __init__(self, alpha=None, gamma=2.0, weight_ce_or_focal=0.6, weight_dice=0.4, mode='ce'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight_ce_or_focal = weight_ce_or_focal
        self.weight_dice = weight_dice
        self.mode = mode
        
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        if self.mode == 'focal':
            base_loss = self.focal(inputs, targets)
        else:
            # Weighted CrossEntropy Loss
            log_softmax = F.log_softmax(inputs, dim=1)
            base_loss = F.nll_loss(log_softmax, targets, weight=self.alpha)
            
        dice_loss = self.dice(inputs, targets)
        return self.weight_ce_or_focal * base_loss + self.weight_dice * dice_loss



def get_segformer_model(pretrained=True):
    """
    Creates and returns a SegFormer model based on config.SEGFORMER_BACKBONE.
    """
    if pretrained:
        print(f"Loading pretrained SegFormer backbone: {config.SEGFORMER_BACKBONE}")
        model = SegformerForSemanticSegmentation.from_pretrained(
            config.SEGFORMER_BACKBONE,
            num_labels=config.NUM_CLASSES,
            ignore_mismatched_sizes=True,
            use_safetensors=True
        )
    else:
        print(f"Initializing new SegFormer configuration for: {config.SEGFORMER_BACKBONE}...")
        configuration = SegformerConfig.from_pretrained(
            config.SEGFORMER_BACKBONE,
            num_labels=config.NUM_CLASSES
        )
        model = SegformerForSemanticSegmentation(configuration)
        
    return model

# Keep alias for backwards compatibility
get_segformer_b3_model = get_segformer_model


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for attention-based feature refinement."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MultiTaskRoadHazardModel(nn.Module):
    """
    Multi-Task Learning Network wrapping SegFormer.
    Outputs:
      1. Segmentation logits (B, NumClasses, H, W)
      2. Severity prediction (B, 3) -> Low, Medium, High
      3. Road Quality score (B, 1) -> [0, 10]
      4. Collision Risk score (B, 1) -> [0, 100]
    """
    def __init__(self, segformer_model, hidden_dim=512):
        super().__init__()
        self.segformer = segformer_model
        
        # Attention module
        self.attention = SEBlock(channels=hidden_dim)
        
        # Severity prediction head
        self.severity_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)
        )
        
        # Road score regression head
        self.road_score_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Risk score prediction head (takes pooled features + physical inputs: area, depth, distance, speed)
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim + 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, pixel_values, physical_inputs=None):
        # Forward pass through SegFormer backbone
        outputs = self.segformer(pixel_values=pixel_values, output_hidden_states=True)
        logits = outputs.logits  # B x NumClasses x H/4 x W/4
        
        # Extract last stage encoder features
        # Segformer hidden states contains: input embedding + 4 encoder stages
        if outputs.hidden_states is not None and len(outputs.hidden_states) > 0:
            feat = outputs.hidden_states[-1] # B x C x H/32 x W/32
            # Apply attention
            feat = self.attention(feat)
            # Global pooling
            pooled = F.adaptive_avg_pool2d(feat, (1, 1)).squeeze(-1).squeeze(-1)
        else:
            # Fallback to pooling logits
            pooled = F.adaptive_avg_pool2d(logits, (1, 1)).squeeze(-1).squeeze(-1)
            # Pad to match hidden_dim if needed
            if pooled.shape[-1] < 512:
                pad_size = 512 - pooled.shape[-1]
                pooled = F.pad(pooled, (0, pad_size))
                
        severity_logits = self.severity_head(pooled)
        road_score = self.road_score_head(pooled) * 10.0
        
        if physical_inputs is None:
            physical_inputs = torch.zeros(pixel_values.shape[0], 4, device=pixel_values.device)
            
        risk_input = torch.cat([pooled, physical_inputs], dim=-1)
        risk_score = self.risk_head(risk_input) * 100.0
        
        return logits, severity_logits, road_score, risk_score


class EnsembleRoadSegmenter(nn.Module):
    """
    Ensemble model combining SegFormer and DeepLabV3+ predictions.
    """
    def __init__(self, segformer_model, deeplabv3_model=None):
        super().__init__()
        self.segformer = segformer_model
        self.deeplabv3 = deeplabv3_model

    def forward(self, pixel_values):
        outputs_seg = self.segformer(pixel_values=pixel_values)
        logits_seg = outputs_seg.logits
        
        # Upscale SegFormer logits
        logits_seg_upscaled = F.interpolate(
            logits_seg,
            size=pixel_values.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        
        if self.deeplabv3 is not None:
            logits_deep = self.deeplabv3(pixel_values)
            # Average predictions (ensemble)
            return (logits_seg_upscaled + logits_deep) / 2.0
        
        return logits_seg_upscaled

