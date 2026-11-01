"""
calculate_correlation_heatmaps.py
=================================
Calculates cross-level annotation correlation (or co-occurrence) heatmaps
between:
  1. Emotion vs Rest (Target, Irony, Hate, Text-Image Relations)
  2. Irony vs Rest (Target, Hate, Emotions, Text-Image Relations)
  3. Semiotic Relation vs Rest (Target, Hate, Emotions, Irony)

Supports:
  - Multimodal Union (Train + Test combined, N=645)
  - Multimodal Train (N=516)
  - Multimodal Test (N=129)
  - Textual Only (N=2023, semiotic relation level automatically omitted)

Outputs publication-quality heatmaps identical in style to:
  - emotion_correlations_english.png
  - irony_correlations_english.png
  - relation_correlations_english.png
  - correlations_combined_banner.png
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Publication typography setup
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['mathtext.fontset'] = 'stix'

# Mapping Italian dataset column names to English publication standard
LABEL_MAPPING = {
    # Emotions
    'Amore': 'Love',
    'Anticipazione': 'Anticipation',
    'Disgusto': 'Disgust',
    'Fiducia': 'Trust',
    'Gioia': 'Joy',
    'Paura': 'Fear',
    'Rabbia': 'Anger',
    'Sorpresa': 'Surprise',
    'Tristezza': 'Sadness',
    'Neutro': 'Neutral',
    
    # Target / Context
    'Contenitore': 'Container',
    'Contenuto': 'Content',
    'Offensivita': 'Offensiveness',
    
    # Irony
    'Ironia': 'Irony',
    'Sarcasmo': 'Sarcasm',
    'None': 'Unironic',
    
    # Text-Image Semiotic Relations
    'Additiva': 'Additive',
    'Parallela': 'Parallel',
    'Divergente': 'Divergent'
}

EMOTION_ROWS = ['Love', 'Anticipation', 'Disgust', 'Trust', 'Joy', 'Fear', 'Anger', 'Surprise', 'Sadness', 'Neutral']
IRONY_ROWS = ['Irony', 'Sarcasm', 'Unironic']
RELATION_ROWS = ['Additive', 'Parallel', 'Divergent']


def draw_custom_heatmap(ax, data, rows, cols, title, metric='pearson', vmin=-0.45, vmax=0.45, fontsize=8.5):
    """
    Renders publication-style heatmap matching the reference paper figures.
    """
    if metric == 'pearson':
        cmap = plt.cm.RdBu_r
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    else:
        cmap = plt.cm.Blues
        im = ax.imshow(data, cmap=cmap, aspect='auto')

    # White grid lines between cells
    ax.set_xticks(np.arange(len(cols)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(rows)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    # Major axis ticks and labels
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(cols, rotation=42, ha='right', fontsize=fontsize)
    ax.set_yticklabels(rows, fontsize=fontsize)

    # Numerical cell values
    for i in range(len(rows)):
        for j in range(len(cols)):
            val = data[i, j]
            if np.isnan(val):
                text_str = "N/A"
                color = "gray"
            else:
                if metric == 'pearson':
                    text_str = f"{val:.3f}"
                    color = 'white' if (val < -0.20 or val > 0.20) else 'black'
                else:
                    text_str = f"{int(val)}"
                    v_max = np.nanmax(data)
                    color = 'white' if val > (v_max * 0.55) else 'black'

            ax.text(j, i, text_str, ha='center', va='center', color=color,
                    fontsize=fontsize - 0.5, fontweight='medium')

    ax.set_title(title, fontsize=fontsize + 2.0, fontweight='bold', pad=9)
    return im


def compute_cross_matrix(df, rows, cols, metric='pearson'):
    """
    Computes either Pearson correlation matrix or joint co-occurrence count matrix.
    """
    if metric == 'pearson':
        all_cols = list(dict.fromkeys(rows + cols))
        corr_matrix = df[all_cols].corr('pearson')
        return corr_matrix.loc[rows, cols]
    else:
        # Co-occurrence counts: C_ij = sum(X_i * X_j)
        return df[rows].T.dot(df[cols])


def generate_heatmaps_for_dataset(csv_or_df, dataset_name, output_dir, prefix="", metric='pearson'):
    """
    Calculates cross-level correlations and generates heatmaps for a dataset.
    Automatically omits semiotic relation levels if not present in the dataframe.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_prefix = f"{prefix}_" if prefix else ""

    if isinstance(csv_or_df, pd.DataFrame):
        df = csv_or_df.copy()
        print(f"\nProcessing: {dataset_name} ({len(df)} rows)")
    else:
        if not os.path.exists(csv_or_df):
            print(f"Error: {csv_or_df} does not exist.")
            return
        print(f"\nProcessing: {csv_or_df}")
        df = pd.read_csv(csv_or_df)

    # Rename existing columns to English standard
    df_renamed = df.rename(columns=LABEL_MAPPING)
    has_relations = 'Additive' in df_renamed.columns

    # -----------------------------------------------------------------------
    # 1. EMOTION LEVEL (Emotion vs Rest)
    # -----------------------------------------------------------------------
    emo_cols = ['Container', 'Content', 'Irony', 'Sarcasm', 'Offensiveness']
    if has_relations:
        emo_cols += ['Additive', 'Divergent', 'Parallel']

    available_emo_rows = [r for r in EMOTION_ROWS if r in df_renamed.columns]
    available_emo_cols = [c for c in emo_cols if c in df_renamed.columns]

    emo_matrix = compute_cross_matrix(df_renamed, available_emo_rows, available_emo_cols, metric=metric)
    emo_csv_path = os.path.join(output_dir, f"{file_prefix}emotion_correlations.csv")
    emo_matrix.to_csv(emo_csv_path)

    # Plot Emotion Heatmap
    fig_w = 5.2 if has_relations else 4.2
    fig, ax = plt.subplots(figsize=(fig_w, 5.2), dpi=300)
    im = draw_custom_heatmap(ax, emo_matrix.values, available_emo_rows, available_emo_cols,
                             'Emotions Correlations', metric=metric, fontsize=9.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar_label = 'Pearson Correlation ($r$)' if metric == 'pearson' else 'Co-occurrence Count'
    cbar.set_label(cbar_label, fontsize=9.5, labelpad=8)
    cbar.ax.tick_params(labelsize=8.5)
    plt.tight_layout()
    emo_img_path = os.path.join(output_dir, f"{file_prefix}emotion_correlations_english.png")
    plt.savefig(emo_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [+] Saved Emotion heatmap  -> {emo_img_path}")

    # -----------------------------------------------------------------------
    # 2. IRONY LEVEL (Irony vs Rest)
    # -----------------------------------------------------------------------
    irony_cols = ['Container', 'Content', 'Offensiveness'] + EMOTION_ROWS
    if has_relations:
        irony_cols += ['Additive', 'Divergent', 'Parallel']

    available_irony_rows = [r for r in IRONY_ROWS if r in df_renamed.columns]
    available_irony_cols = [c for c in irony_cols if c in df_renamed.columns]

    irony_matrix = compute_cross_matrix(df_renamed, available_irony_rows, available_irony_cols, metric=metric)
    irony_csv_path = os.path.join(output_dir, f"{file_prefix}irony_correlations.csv")
    irony_matrix.to_csv(irony_csv_path)

    # Plot Irony Heatmap
    fig_w = 9.6 if has_relations else 8.0
    fig, ax = plt.subplots(figsize=(fig_w, 2.15), dpi=300)
    im = draw_custom_heatmap(ax, irony_matrix.values, available_irony_rows, available_irony_cols,
                             'Irony & Sarcasm Correlations', metric=metric, fontsize=8.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.025)
    cbar.set_label(cbar_label, fontsize=9.0, labelpad=6)
    cbar.ax.tick_params(labelsize=8.0)
    plt.tight_layout()
    irony_img_path = os.path.join(output_dir, f"{file_prefix}irony_correlations_english.png")
    plt.savefig(irony_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [+] Saved Irony heatmap    -> {irony_img_path}")

    # -----------------------------------------------------------------------
    # 3. TEXT-IMAGE RELATION LEVEL (Relation vs Rest)
    # -----------------------------------------------------------------------
    rel_matrix = None
    if has_relations:
        rel_cols = ['Container', 'Content', 'Offensiveness'] + EMOTION_ROWS + ['Irony', 'Sarcasm']
        available_rel_rows = [r for r in RELATION_ROWS if r in df_renamed.columns]
        available_rel_cols = [c for c in rel_cols if c in df_renamed.columns]

        rel_matrix = compute_cross_matrix(df_renamed, available_rel_rows, available_rel_cols, metric=metric)
        rel_csv_path = os.path.join(output_dir, f"{file_prefix}relation_correlations.csv")
        rel_matrix.to_csv(rel_csv_path)

        # Plot Relation Heatmap
        fig, ax = plt.subplots(figsize=(9.2, 2.15), dpi=300)
        im = draw_custom_heatmap(ax, rel_matrix.values, available_rel_rows, available_rel_cols,
                                 'Semiotic Relation Correlations', metric=metric, fontsize=8.5)
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.025)
        cbar.set_label(cbar_label, fontsize=9.0, labelpad=6)
        cbar.ax.tick_params(labelsize=8.0)
        plt.tight_layout()
        rel_img_path = os.path.join(output_dir, f"{file_prefix}relation_correlations_english.png")
        plt.savefig(rel_img_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [+] Saved Relation heatmap -> {rel_img_path}")
    else:
        print("  [-] Semiotic Relation level omitted (not present in textual dataset)")

    # -----------------------------------------------------------------------
    # 4. COMBINED PUBLICATION BANNER (for multimodal datasets)
    # -----------------------------------------------------------------------
    if has_relations and rel_matrix is not None and metric == 'pearson':
        fig = plt.figure(figsize=(10.5, 4.4), dpi=300)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.45], height_ratios=[1, 1], wspace=0.25, hspace=0.48)

        # (a) Emotion correlations (left column)
        ax0 = fig.add_subplot(gs[:, 0])
        draw_custom_heatmap(ax0, emo_matrix.values, available_emo_rows, available_emo_cols,
                            '(a) Emotion Correlations with Context', metric=metric, fontsize=8.0)

        # (b) Irony correlations (top right)
        ax1 = fig.add_subplot(gs[0, 1])
        draw_custom_heatmap(ax1, irony_matrix.values, available_irony_rows, available_irony_cols,
                            '(b) Irony & Sarcasm Correlations', metric=metric, fontsize=7.5)

        # (c) Semiotic relation correlations (bottom right)
        ax2 = fig.add_subplot(gs[1, 1])
        im_banner = draw_custom_heatmap(ax2, rel_matrix.values, available_rel_rows, available_rel_cols,
                                        '(c) Semiotic Relation Correlations', metric=metric, fontsize=7.5)

        # Colorbar on the far right
        cbar_ax = fig.add_axes([0.935, 0.18, 0.015, 0.65])
        cbar_b = fig.colorbar(im_banner, cax=cbar_ax)
        cbar_b.set_label('Pearson Correlation ($r$)', fontsize=9.0, labelpad=8)
        cbar_b.ax.tick_params(labelsize=8.0)

        banner_path = os.path.join(output_dir, f"{file_prefix}correlations_combined_banner.png")
        plt.savefig(banner_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [+] Saved Combined banner  -> {banner_path}")


def main():
    parser = argparse.ArgumentParser(description="Calculate cross-level annotation correlation / co-occurrence heatmaps.")
    parser.add_argument('--dataset', type=str, default=None,
                        help="Path to specific CSV file. If omitted, runs on all datasets and multimodal union.")
    parser.add_argument('--output-dir', type=str, default=None,
                        help="Directory to save output heatmaps and CSVs.")
    parser.add_argument('--metric', type=str, default='pearson', choices=['pearson', 'cooccurrence'],
                        help="Metric to compute: 'pearson' (correlation matching paper figures) or 'cooccurrence' (joint counts).")
    parser.add_argument('--prefix', type=str, default="",
                        help="Prefix for output filenames (default: empty).")
    parser.add_argument('--union-only', action='store_true',
                        help="Only compute heatmaps on the multimodal union (train + test, N=645).")
    args = parser.parse_args()

    default_dir = os.path.dirname(os.path.abspath(__file__))
    base_output_dir = args.output_dir if args.output_dir else os.path.join(default_dir, 'heatmaps')

    # Paths to the multimodal partitions
    train_csv = os.path.join(default_dir, 'multimodal_train_seed42_516.csv')
    test_csv = os.path.join(default_dir, 'multimodal_test_seed42_129.csv')
    textual_csv = os.path.join(default_dir, 'textual_only_cleaned_2023.csv')

    # Fallback paths without count suffix
    if not os.path.exists(train_csv):
        train_csv = os.path.join(default_dir, 'multimodal_train_seed42.csv')
    if not os.path.exists(test_csv):
        test_csv = os.path.join(default_dir, 'multimodal_test_seed42.csv')
    if not os.path.exists(textual_csv):
        textual_csv = os.path.join(default_dir, 'textual_only_cleaned.csv')

    # Build Multimodal Union (N = 645)
    union_df = None
    if os.path.exists(train_csv) and os.path.exists(test_csv):
        df_tr = pd.read_csv(train_csv)
        df_te = pd.read_csv(test_csv)
        union_df = pd.concat([df_tr, df_te], ignore_index=True)
        # Save union CSV for convenience if not already present
        union_csv_path = os.path.join(default_dir, 'multimodal_union_645.csv')
        if not os.path.exists(union_csv_path):
            union_df.to_csv(union_csv_path, index=False)

    if args.dataset:
        dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
        generate_heatmaps_for_dataset(args.dataset, dataset_name, output_dir=base_output_dir,
                                      prefix=args.prefix, metric=args.metric)
    elif args.union_only:
        if union_df is not None:
            union_out_dir = os.path.join(base_output_dir, 'multimodal_union')
            generate_heatmaps_for_dataset(union_df, 'Multimodal Union (N=645)',
                                          output_dir=union_out_dir, prefix=args.prefix, metric=args.metric)
        else:
            print("Error: Could not locate multimodal train and test datasets to form union.")
    else:
        # 1. Multimodal Union (N = 645)
        if union_df is not None:
            union_out_dir = os.path.join(base_output_dir, 'multimodal_union')
            generate_heatmaps_for_dataset(union_df, 'Multimodal Union (N=645)',
                                          output_dir=union_out_dir, prefix="", metric=args.metric)

        # 2. Multimodal Train (N = 516)
        if os.path.exists(train_csv):
            train_out_dir = os.path.join(base_output_dir, 'multimodal_train')
            generate_heatmaps_for_dataset(train_csv, 'Multimodal Train (N=516)',
                                          output_dir=train_out_dir, prefix="", metric=args.metric)

        # 3. Multimodal Test (N = 129)
        if os.path.exists(test_csv):
            test_out_dir = os.path.join(base_output_dir, 'multimodal_test')
            generate_heatmaps_for_dataset(test_csv, 'Multimodal Test (N=129)',
                                          output_dir=test_out_dir, prefix="", metric=args.metric)

        # 4. Textual Only (N = 2023)
        if os.path.exists(textual_csv):
            textual_out_dir = os.path.join(base_output_dir, 'textual_only')
            generate_heatmaps_for_dataset(textual_csv, 'Textual Only (N=2023)',
                                          output_dir=textual_out_dir, prefix="", metric=args.metric)


if __name__ == '__main__':
    main()

