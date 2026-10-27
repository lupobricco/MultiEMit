import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
imgs_dir = os.path.join(script_dir, 'imgs')
os.makedirs(imgs_dir, exist_ok=True)

# Set publication style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['mathtext.fontset'] = 'stix'

# Exact values extracted from user uploaded images

# 1. EMOTION CORRELATIONS (10 emotions x 8 context variables)
emo_rows = ['Love', 'Anticipation', 'Disgust', 'Trust', 'Joy', 'Fear', 'Anger', 'Surprise', 'Sadness', 'Neutral']
emo_cols = ['Container', 'Content', 'Irony', 'Sarcasm', 'Offensiveness', 'Additive', 'Divergent', 'Parallel']

emo_data = np.array([
    [ 0.088, -0.093, -0.100, -0.052, -0.028,  0.081, -0.111,  0.020],
    [ 0.181, -0.159, -0.061, -0.058, -0.031,  0.111, -0.073, -0.065],
    [-0.203,  0.205, -0.002,  0.400,  0.212, -0.039,  0.165, -0.146],
    [ 0.229, -0.178, -0.247, -0.126, -0.068,  0.121, -0.264,  0.162],
    [-0.079,  0.108,  0.185, -0.106, -0.058, -0.017,  0.106, -0.098],
    [-0.099,  0.118, -0.036, -0.014, -0.008,  0.064, -0.045, -0.034],
    [-0.083,  0.098, -0.088,  0.158,  0.266,  0.064,  0.011, -0.101],
    [-0.270,  0.319,  0.176, -0.015, -0.037, -0.090,  0.194, -0.112],
    [-0.153,  0.214, -0.091,  0.010, -0.019,  0.146, -0.096, -0.086],
    [-0.102, -0.017,  0.021, -0.072, -0.039, -0.057, -0.000,  0.075]
])

# 2. IRONY CORRELATIONS (3 rows x 16 columns)
irony_rows = ['Irony', 'Sarcasm', 'Unironic']
irony_cols = ['Container', 'Content', 'Offensiveness', 'Love', 'Anticipation', 'Disgust', 
              'Trust', 'Joy', 'Fear', 'Anger', 'Surprise', 'Sadness', 'Neutral', 
              'Additive', 'Divergent', 'Parallel']

irony_data = np.array([
    [-0.312,  0.272, -0.044, -0.100, -0.061, -0.002, -0.247,  0.185, -0.036, -0.088,  0.176, -0.091,  0.021, -0.251,  0.407, -0.147],
    [-0.005,  0.016,  0.169, -0.052, -0.058,  0.400, -0.126, -0.106, -0.014,  0.158, -0.015,  0.010, -0.072, -0.056,  0.127, -0.077],
    [ 0.294, -0.262, -0.032,  0.116,  0.082, -0.171,  0.286, -0.127,  0.039,  0.014, -0.158,  0.080,  0.011,  0.260, -0.436,  0.171]
])

# 3. SEMIOTIC RELATION CORRELATIONS (3 rows x 15 columns)
rel_rows = ['Additive', 'Parallel', 'Divergent']
rel_cols = ['Container', 'Content', 'Offensiveness', 'Love', 'Anticipation', 'Disgust', 
            'Trust', 'Joy', 'Fear', 'Anger', 'Surprise', 'Sadness', 'Neutral', 
            'Irony', 'Sarcasm']

rel_data = np.array([
    [ 0.194, -0.117,  0.046,  0.081,  0.111, -0.039,  0.121, -0.017,  0.064,  0.064, -0.090,  0.146, -0.057, -0.251, -0.056],
    [ 0.112, -0.138, -0.042,  0.020, -0.065, -0.146,  0.162, -0.098, -0.034, -0.101, -0.112, -0.086,  0.075, -0.147, -0.077],
    [-0.316,  0.252, -0.018, -0.111, -0.073,  0.165, -0.264,  0.106, -0.045,  0.011,  0.194, -0.096, -0.000,  0.407,  0.127]
])

cmap = plt.cm.RdBu_r

def draw_custom_heatmap(ax, data, rows, cols, title, vmin=-0.45, vmax=0.45, fontsize=8):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    
    # Grid lines
    ax.set_xticks(np.arange(len(cols)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(rows)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)
    
    # Major ticks
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(cols, rotation=42, ha='right', fontsize=fontsize)
    ax.set_yticklabels(rows, fontsize=fontsize)
    
    # Value annotations
    for i in range(len(rows)):
        for j in range(len(cols)):
            val = data[i, j]
            # White text for high saturation cells
            color = 'white' if (val < -0.20 or val > 0.20) else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', color=color, 
                    fontsize=fontsize-0.5, fontweight='medium')
            
    ax.set_title(title, fontsize=fontsize+1.5, fontweight='bold', pad=8)
    return im

# -------------------------------------------------------------------
# COMBINED BANNER (Full width figure* for Page 4)
# -------------------------------------------------------------------
fig = plt.figure(figsize=(10.5, 4.4), dpi=300)
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.45], height_ratios=[1, 1], wspace=0.25, hspace=0.48)

# (a) Emotion correlations (spans both rows on the left)
ax0 = fig.add_subplot(gs[:, 0])
draw_custom_heatmap(ax0, emo_data, emo_rows, emo_cols, '(a) Emotion Correlations with Context', fontsize=8)

# (b) Irony correlations (top right)
ax1 = fig.add_subplot(gs[0, 1])
draw_custom_heatmap(ax1, irony_data, irony_rows, irony_cols, '(b) Irony & Sarcasm Correlations', fontsize=7.5)

# (c) Semiotic relation correlations (bottom right)
ax2 = fig.add_subplot(gs[1, 1])
im = draw_custom_heatmap(ax2, rel_data, rel_rows, rel_cols, '(c) Semiotic Relation Correlations', fontsize=7.5)

# Colorbar on the far right
cbar_ax = fig.add_axes([0.935, 0.18, 0.015, 0.65])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Pearson Correlation ($r$)', fontsize=9, labelpad=8)
cbar.ax.tick_params(labelsize=8)

plt.savefig(os.path.join(imgs_dir, 'correlations_combined_banner.png'), dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------
# INDIVIDUAL FIGURES (For standalone inclusion)
# -------------------------------------------------------------------

# 1. Emotions Correlations
fig, ax = plt.subplots(figsize=(5.0, 5.2), dpi=300)
im = draw_custom_heatmap(ax, emo_data, emo_rows, emo_cols, 'Emotions Correlations', fontsize=9.5)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Pearson Correlation ($r$)', fontsize=9.5, labelpad=8)
cbar.ax.tick_params(labelsize=8.5)
plt.tight_layout()
plt.savefig(os.path.join(imgs_dir, 'emotion_correlations_english.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Irony Correlations (Wide page-width format)
fig, ax = plt.subplots(figsize=(9.6, 2.15), dpi=300)
im = draw_custom_heatmap(ax, irony_data, irony_rows, irony_cols, 'Irony & Sarcasm Correlations', fontsize=9.0)
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.025)
cbar.set_label('Pearson Correlation ($r$)', fontsize=9.0, labelpad=6)
cbar.ax.tick_params(labelsize=8.0)
plt.tight_layout()
plt.savefig(os.path.join(imgs_dir, 'irony_correlations_english.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Semiotic Relation Correlations (Wide page-width format)
fig, ax = plt.subplots(figsize=(9.2, 2.15), dpi=300)
im = draw_custom_heatmap(ax, rel_data, rel_rows, rel_cols, 'Semiotic Relation Correlations', fontsize=9.0)
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.025)
cbar.set_label('Pearson Correlation ($r$)', fontsize=9.0, labelpad=6)
cbar.ax.tick_params(labelsize=8.0)
plt.tight_layout()
plt.savefig(os.path.join(imgs_dir, 'relation_correlations_english.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Recreated all heatmaps with 'Emotions Correlations' title, larger readable fonts, and separate high-res images!")

