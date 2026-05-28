#!/usr/bin/env python3
"""Test the trained CloudCNN and produce a CLEAR comparison image."""
import os, sys, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import path_setup
ROOT = path_setup.root_path()

from cloud_cnn import CloudCNN
from modis_real_data import MODISCloudPatchDataset

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(ROOT, "models/cloud_cnn_real.pt"))
ap.add_argument("--data-dir", default=os.path.join(ROOT, "data/modis_test"))
ap.add_argument("--cloud", default=os.path.join(ROOT, "config/cloud_reality/algeria_real_clouds.json"))
ap.add_argument("--n-test", type=int, default=2000)    # use more samples
ap.add_argument("--seed", type=int, default=123)
ap.add_argument("--save", action="store_true")
ap.add_argument("--out", default="cloud_cnn_test_comparison.png")
args = ap.parse_args()

# Load model
model = CloudCNN()
model.load_state_dict(torch.load(args.model, map_location='cpu'))
model.eval()

# Test dataset (real only)
dataset = MODISCloudPatchDataset(
    args.cloud, n_samples=args.n_test,
    real_patch_dir=args.data_dir, real_mix_ratio=1.0, seed=args.seed)
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

preds, labels = [], []
with torch.no_grad():
    for patches, lbls in loader:
        out = model(patches).squeeze(1)
        preds.append(out.numpy())
        labels.append(lbls.squeeze(1).numpy())

preds = np.concatenate(preds)
labels = np.concatenate(labels)

rmse = np.sqrt(np.mean((preds - labels)**2))
mae  = np.mean(np.abs(preds - labels))
acc  = np.mean((preds > 0.5) == (labels > 0.5))
print(f"Test set ({len(preds)} patches):")
print(f"  RMSE: {rmse:.4f}  |  MAE: {mae:.4f}  |  Accuracy: {acc:.2%}")

# --- VISUALIZATION ---
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# 1. Example patches with clear TRUE / PRED labels
for i in range(3):
    patch, cf_true = dataset[i]       # dataset returns (tensor, tensor)
    cf_true_val = cf_true.item() if hasattr(cf_true, 'item') else float(cf_true)
    with torch.no_grad():
        cf_pred_val = model(patch.unsqueeze(0)).item()

    ax = axes[0, i]
    ax.imshow(patch.numpy().transpose(1, 2, 0))
    ax.set_title(f"TRUE cloud fraction: {cf_true_val:.2f}\nPREDICTED cloud fraction: {cf_pred_val:.2f}",
                 fontsize=12, fontweight='bold', color='darkgreen')
    ax.axis('off')

# 2. Scatter plot with identity line
ax_scatter = axes[1, 0]
ax_scatter.scatter(labels, preds, alpha=0.3, s=2, color='steelblue')
ax_scatter.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect prediction')
ax_scatter.set_xlabel('True cloud fraction', fontsize=11)
ax_scatter.set_ylabel('Predicted cloud fraction', fontsize=11)
ax_scatter.set_title(f'All {len(preds)} test samples\nRMSE={rmse:.3f}  Acc={acc:.2%}', fontsize=11)
ax_scatter.legend()
ax_scatter.grid(True, alpha=0.3)

# 3. Error histogram
ax_hist = axes[1, 1]
errors = preds - labels
ax_hist.hist(errors, bins=50, color='steelblue', edgecolor='white')
ax_hist.set_xlabel('Prediction error (pred - true)', fontsize=11)
ax_hist.set_ylabel('Number of patches', fontsize=11)
ax_hist.set_title('Error distribution', fontsize=11)
ax_hist.grid(axis='y', alpha=0.3)

# 4. Summary text
ax_text = axes[1, 2]
ax_text.axis('off')
summary = (
    f"Model: cloud_cnn_real.pt\n"
    f"Test samples: {len(preds)}\n"
    f"RMSE: {rmse:.4f}\n"
    f"MAE : {mae:.4f}\n"
    f"Cloud detection accuracy\n(at 0.5 threshold):\n{acc:.2%}\n\n"
    "Blue dots = one patch\n"
    "Red line = perfect match"
)
ax_text.text(0.05, 0.95, summary, fontsize=11, verticalalignment='top',
             family='monospace', transform=ax_text.transAxes)

plt.suptitle("Cloud CNN — Test on Independent MODIS Patches", fontsize=14, y=0.98)
plt.tight_layout()

if args.save:
    plt.savefig(args.out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Figure saved → {args.out}")
else:
    plt.show()