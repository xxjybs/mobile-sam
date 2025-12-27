"""
Example script demonstrating how to use SmallDefectLoss for training
针对密集小缺陷检测的损失函数使用示例
"""

import torch
from helper.loss import SmallDefectLoss, FocalLoss, TverskyLoss

# ========== Example 1: Basic usage of SmallDefectLoss ==========
print("=" * 60)
print("Example 1: Basic usage of SmallDefectLoss")
print("=" * 60)

# Create loss function with default parameters optimized for small defects
criterion = SmallDefectLoss()

# Simulate model predictions and ground truth
batch_size = 2
height, width = 256, 256

# Model output (logits, before sigmoid)
pred_logits = torch.randn(batch_size, 1, height, width)

# Ground truth (binary mask: 0 for background, 1 for defect)
target_mask = torch.zeros(batch_size, 1, height, width)
# Add some small defects
target_mask[:, :, 100:110, 100:110] = 1  # 10x10 small defect
target_mask[:, :, 150:155, 150:155] = 1  # 5x5 tiny defect

# Calculate loss
loss = criterion(pred_logits, target_mask)
print(f"SmallDefectLoss: {loss.item():.4f}")
print()

# ========== Example 2: Customized parameters ==========
print("=" * 60)
print("Example 2: Customized parameters for different scenarios")
print("=" * 60)

# For extremely dense tiny defects - increase recall focus
criterion_high_recall = SmallDefectLoss(
    focal_weight=1.0,
    tversky_weight=3.0,  # Increase Tversky weight
    iou_weight=0.5,      # Decrease IOU weight
    tversky_alpha=0.2,   # Lower alpha (less penalty for false positives)
    tversky_beta=0.8     # Higher beta (more penalty for false negatives)
)

loss_high_recall = criterion_high_recall(pred_logits, target_mask)
print(f"High Recall Loss: {loss_high_recall.item():.4f}")

# For balanced performance
criterion_balanced = SmallDefectLoss(
    focal_weight=1.0,
    tversky_weight=1.5,
    iou_weight=1.0,
    tversky_alpha=0.4,
    tversky_beta=0.6
)

loss_balanced = criterion_balanced(pred_logits, target_mask)
print(f"Balanced Loss: {loss_balanced.item():.4f}")
print()

# ========== Example 3: Individual loss components ==========
print("=" * 60)
print("Example 3: Understanding individual loss components")
print("=" * 60)

focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
tversky_loss = TverskyLoss(alpha=0.3, beta=0.7)

# Calculate individual losses
loss_focal = focal_loss(pred_logits, target_mask)
loss_tversky = tversky_loss(pred_logits, target_mask)

print(f"Focal Loss:   {loss_focal.item():.4f}")
print(f"Tversky Loss: {loss_tversky.item():.4f}")
print(f"Combined:     {(loss_focal + 2*loss_tversky).item():.4f}")
print()

# ========== Example 4: Training loop integration ==========
print("=" * 60)
print("Example 4: Integration in training loop")
print("=" * 60)

print("""
# In your training script:

from helper.loss import SmallDefectLoss

# Create loss function
criterion = SmallDefectLoss(
    focal_weight=1.0,
    tversky_weight=2.0,
    iou_weight=1.0,
    tversky_alpha=0.3,
    tversky_beta=0.7
)

# Move to GPU if available
if torch.cuda.is_available():
    criterion.cuda()

# In training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        images, masks = batch
        
        # Forward pass
        predictions = model(images)
        
        # Calculate loss
        loss = criterion(predictions, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
""")

print()
print("=" * 60)
print("Parameter Guidelines for Different Scenarios:")
print("=" * 60)
print("""
Scenario 1: Dense tiny defects (like 398.png)
- tversky_weight: 2.0-3.0 (emphasize recall)
- tversky_alpha: 0.2-0.3 (tolerate false positives)
- tversky_beta: 0.7-0.8 (penalize false negatives heavily)
- focal_gamma: 2.0-3.0 (focus on hard examples)

Scenario 2: Sparse but clear defects
- tversky_weight: 1.0-1.5 (balanced)
- tversky_alpha: 0.5
- tversky_beta: 0.5
- focal_gamma: 2.0

Scenario 3: Mixed defect sizes
- tversky_weight: 1.5-2.0
- tversky_alpha: 0.3-0.4
- tversky_beta: 0.6-0.7
- focal_gamma: 2.0

Key Principles:
- Lower tversky_alpha → More aggressive detection (higher recall, possibly lower precision)
- Higher tversky_beta → More penalty for missing defects
- Higher focal_gamma → More focus on hard/small objects
- Higher tversky_weight → More emphasis on recall vs overall segmentation quality
""")
