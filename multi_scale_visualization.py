import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import os
from PIL import Image

from model.load_datasets import load_datasets


# -----------------------------
# Configuration
# -----------------------------
dataset = '3D_concentric_circles'
T_list = [0, 9, 19, 29, 39, 49, 59, 69, 79]

# Label mapping
label_mapping = {
    '3D_concentric_circles': {
        0: "Circle_1",
        1: "Circle_2",
        2: "Circle_3",
        3: "Circle_4"
    }
}

# Load dataset
X, y = load_datasets(dataset)
unique_labels = np.unique(y)

# Color palette (consistent across plots and GIF)
colors = [
    "#9467bd",  # deep purple
    "#2ca02c",  # deep green
    "#d62728",  # deep red
    "#1f77b4"   # deep blue
]

# -----------------------------
# Load pNMF embeddings
# -----------------------------
results = scipy.io.loadmat(f'results/{dataset}/pNMF.mat')
H_matrices = results['pNMF']

# -----------------------------
# Determine coordinate ranges for scatter plots
# -----------------------------
all_H = np.hstack(H_matrices)  # stack horizontally
x_min, x_max = all_H[0, :].min(), all_H[0, :].max()
y_min, y_max = all_H[1, :].min(), all_H[1, :].max()
x_pad = (x_max - x_min) * 0.1
y_pad = (y_max - y_min) * 0.1
x_range = (x_min - x_pad, x_max + x_pad)
y_range = (y_min - y_pad, y_max + y_pad)

# -----------------------------
# Multi-scale PDF visualization
# -----------------------------
fig = plt.figure(figsize=(20, 9), dpi=300)
gs = fig.add_gridspec(2, 5)

# Original 3D data
ax = fig.add_subplot(gs[0, 0], projection='3d')
ax.scatter(X.T[:, 0], X.T[:, 1], X.T[:, 2],
           color=[colors[k % len(unique_labels)] for k in y], alpha=1, s=20)
ax.set_title(dataset, fontsize=20)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.xaxis.set_pane_color((1, 1, 1, 1))
ax.yaxis.set_pane_color((1, 1, 1, 1))
ax.zaxis.set_pane_color((1, 1, 1, 1))

# Plot embeddings at selected time steps
for idx, t in enumerate(T_list):
    row = (idx + 1) // 5
    col = (idx + 1) % 5
    ax = fig.add_subplot(gs[row, col])
    H = H_matrices[t]
    ax.scatter(H.T[:, 0], H.T[:, 1], color=[colors[k % len(unique_labels)] for k in y], alpha=1, s=20)
    ax.set_title(f'$H_{{{t+1}}}$', fontsize=24)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xticks([])
    ax.set_yticks([])

# Add legend in the bottom center
handles = [plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=colors[k], markersize=12)
           for k in unique_labels]
labels = [label_mapping[dataset][k] for k in unique_labels]
fig.legend(handles, labels, loc='lower center', ncol=len(unique_labels),
           fontsize=24, frameon=False, bbox_to_anchor=(0.5, 0))

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(f'results/{dataset}/{dataset}_pNMF_multi_scale_visualization.pdf', dpi=300)
plt.show()

# -----------------------------
# GIF visualization
# -----------------------------
temp_folder = f'results/{dataset}/multi_scale_visualization'
os.makedirs(temp_folder, exist_ok=True)
image_files = []

# Save each frame
for i in range(len(H_matrices)):
    H = H_matrices[len(H_matrices)-1-i]
    plt.figure(figsize=(4, 4), dpi=300)
    plt.scatter(H.T[:, 0], H.T[:, 1], color=[colors[k % len(unique_labels)] for k in y], alpha=1, s=20)
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.axis("off")
    plt.title(f'$H_{{{len(H_matrices)-i}}}$')
    
    img_path = os.path.join(temp_folder, f'H_{len(H_matrices)-i}.png')
    plt.savefig(img_path)
    plt.close()
    image_files.append(img_path)

# Combine frames into GIF
images = [Image.open(img) for img in image_files]
gif_name = f'results/{dataset}/animation_{dataset}_pNMF.gif'
images[0].save(gif_name, save_all=True, append_images=images[1:], loop=0, duration=100)
print(f"GIF animation saved to {gif_name}")