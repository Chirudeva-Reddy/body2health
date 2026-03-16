import torch
import torch.nn.functional as F
import numpy as np
import cv2

def extract_bbox_features(mask: torch.Tensor, crop_head_ratio: float = 0.15) -> torch.Tensor:
    """
    Extract normalized bbox features from a binary mask.
    mask: (B, 1, H, W) float or uint8 binary silhouette.
    Returns (B, 4): normalized height, width, area, aspect (w/h) after optional head cropping.
    """
    B, _, H, W = mask.shape
    device = mask.device
    features = []
    for i in range(B):
        m = mask[i, 0].cpu().numpy()  # (H, W)
        # Optionally crop top portion (head) to ignore it for body shape
        if crop_head_ratio > 0.0:
            crop_h = int(H * crop_head_ratio)
            m = m[crop_h:, :]
        # Find bounding box of non-zero pixels
        rows = np.any(m, axis=1)
        cols = np.any(m, axis=0)
        if not np.any(rows) or not np.any(cols):
            # Empty mask; return zeros
            features.append([0.0, 0.0, 0.0, 0.0])
            continue
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        bbox_h = y_max - y_min + 1
        bbox_w = x_max - x_min + 1
        area = bbox_h * bbox_w
        aspect = bbox_w / (bbox_h + 1e-6)
        # Normalize by image dimensions
        norm_h = bbox_h / H
        norm_w = bbox_w / W
        norm_area = area / (H * W)
        features.append([norm_h, norm_w, norm_area, aspect])
    return torch.tensor(features, dtype=torch.float32, device=device)

def extract_dual_bbox_features(front_mask: torch.Tensor, side_mask: torch.Tensor, crop_head_ratio: float = 0.15) -> torch.Tensor:
    """
    Extract bbox features from front and side masks and concatenate.
    Returns (B, 8): [front_h, front_w, front_area, front_aspect, side_h, side_w, side_area, side_aspect].
    """
    front_feat = extract_bbox_features(front_mask, crop_head_ratio=crop_head_ratio)  # (B, 4)
    side_feat = extract_bbox_features(side_mask, crop_head_ratio=crop_head_ratio)    # (B, 4)
    return torch.cat([front_feat, side_feat], dim=1)  # (B, 8)
