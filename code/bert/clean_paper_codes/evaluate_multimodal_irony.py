"""
Task 2: Multimodal Irony Evaluation (Table IV & Table VI in Synthesis)
Evaluates Blind Text vs. Dual-Encoder (UmBERTo + ViT-224, with StandardScaler) on mm_test (N=129)
across unprompted and correlation-prompted configurations, including ADD/PAR/DIV subsets.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from common_utils import (
    IRONY_LABELS,
    setup_console,
    seed_everything,
    get_paths,
    row_to_irony_class,
    oracle_prompt,
    create_type_column,
    metric_dict,
    compute_subset_metrics,
    resolve_checkpoint,
    extract_text_features,
    extract_vit_features,
    fit_linear_head,
)


def parse_args():
    parser = argparse.ArgumentParser(description="MultiEMiT Multimodal Irony Evaluation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoints-dir", type=str, default="./checkpoints")
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--vit-model", type=str, default="google/vit-base-patch16-224")
    return parser.parse_args()


def main():
    setup_console()
    args = parse_args()
    seed_everything(args.seed)

    paths = get_paths()
    res_dir = Path(args.results_dir).resolve()
    res_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_tr = pd.read_csv(paths["mm_train"])
    df_te = pd.read_csv(paths["mm_test"])
    y_tr = df_tr.apply(row_to_irony_class, axis=1).values
    y_te = df_te.apply(row_to_irony_class, axis=1).values
    types_te = df_te.apply(create_type_column, axis=1).values

    # Extract ViT Features (Frozen, 768-d)
    print("\nExtracting ViT-224 visual features ...")
    vit_tr_raw = extract_vit_features(df_tr["img_path"].values, paths["images"], args.vit_model, device, args.batch_size)
    vit_te_raw = extract_vit_features(df_te["img_path"].values, paths["images"], args.vit_model, device, args.batch_size)

    # Standardize ViT features
    scaler_img = StandardScaler().fit(vit_tr_raw)
    vit_tr = scaler_img.transform(vit_tr_raw)
    vit_te = scaler_img.transform(vit_te_raw)

    # 1. Step 1 Base (Raw text)
    ckpt_s1 = resolve_checkpoint("irony_textual_no_corr", args.checkpoints_dir)
    print(f"\nExtracting Step 1 text features from: {ckpt_s1} ...")
    s1_txt_tr_raw, _ = extract_text_features(df_tr["text"].values, ckpt_s1, device, batch_size=args.batch_size)
    s1_txt_te_raw, _ = extract_text_features(df_te["text"].values, ckpt_s1, device, batch_size=args.batch_size)
    scaler_s1 = StandardScaler().fit(s1_txt_tr_raw)
    s1_txt_tr = scaler_s1.transform(s1_txt_tr_raw)
    s1_txt_te = scaler_s1.transform(s1_txt_te_raw)

    # 2. Step 2 Oracle (Correlated text)
    ckpt_s2 = resolve_checkpoint("irony_textual_corr", args.checkpoints_dir)
    print(f"\nExtracting Step 2 Oracle text features from: {ckpt_s2} ...")
    s2_prompt_tr = oracle_prompt(df_tr, "irony", with_relation=False)
    s2_prompt_te = oracle_prompt(df_te, "irony", with_relation=False)
    s2_txt_tr_raw, _ = extract_text_features(s2_prompt_tr, ckpt_s2, device, batch_size=args.batch_size)
    s2_txt_te_raw, _ = extract_text_features(s2_prompt_te, ckpt_s2, device, batch_size=args.batch_size)
    scaler_s2 = StandardScaler().fit(s2_txt_tr_raw)
    s2_txt_tr = scaler_s2.transform(s2_txt_tr_raw)
    s2_txt_te = scaler_s2.transform(s2_txt_te_raw)

    experiments = [
        {
            "regime": "Step 1 Base (No corr, std)",
            "prompt": "None (Raw)",
            "txt_tr": s1_txt_tr,
            "txt_te": s1_txt_te,
        },
        {
            "regime": "Step 2 Oracle (With corr, std)",
            "prompt": "Oracle",
            "txt_tr": s2_txt_tr,
            "txt_te": s2_txt_te,
        },
    ]

    baseline_rows = []
    subset_rows = []

    for exp in experiments:
        txt_tr, txt_te = exp["txt_tr"], exp["txt_te"]
        de_tr = np.hstack([txt_tr, vit_tr])
        de_te = np.hstack([txt_te, vit_te])

        # A. Blind Text Head
        clf_blind = fit_linear_head(txt_tr, y_tr, multilabel=False, seed=args.seed)
        pred_blind = clf_blind.predict(txt_te)
        m_blind = metric_dict(y_te, pred_blind)
        baseline_rows.append({
            "Configuration": f"Multimodal test set (without images, {exp['regime']})",
            "Prompting": exp["prompt"],
            "Model": "Blind Text (SGD, std)",
            "Macro F1": m_blind["Macro F1"],
            "Micro F1 (= Acc.)": m_blind["Accuracy"],
            "Weighted F1": m_blind["Weighted F1"],
        })

        # B. Dual-Encoder Head
        clf_de = fit_linear_head(de_tr, y_tr, multilabel=False, seed=args.seed)
        pred_de = clf_de.predict(de_te)
        m_de = metric_dict(y_te, pred_de)
        baseline_rows.append({
            "Configuration": f"Multimodal test set with images (Dual-Encoder, {exp['regime']})",
            "Prompting": exp["prompt"],
            "Model": "Dual-Encoder (BERT + ViT-224, std)",
            "Macro F1": m_de["Macro F1"],
            "Micro F1 (= Acc.)": m_de["Accuracy"],
            "Weighted F1": m_de["Weighted F1"],
        })

        # Semiotic Subsets for Dual-Encoder
        sub_m = compute_subset_metrics(y_te, pred_de, types_te, multilabel=False)
        subset_rows.append({
            "Model Setting": f"Dual-Encoder ({exp['regime']})",
            "ADD (N=68)": sub_m["ADD"]["Accuracy"],
            "PAR (N=23)": sub_m["PAR"]["Accuracy"],
            "DIV (N=38)": sub_m["DIV"]["Accuracy"],
        })

    df_base = pd.DataFrame(baseline_rows)
    df_sub = pd.DataFrame(subset_rows)

    df_base.to_csv(res_dir / "multimodal_irony_baselines.csv", index=False)
    df_sub.to_csv(res_dir / "multimodal_irony_subsets.csv", index=False)

    print("\n" + "=" * 80)
    print("TASK 2: MULTIMODAL IRONY BASELINES (mm_test, N=129) [Table IV]")
    print("=" * 80)
    print(df_base[["Configuration", "Macro F1", "Micro F1 (= Acc.)", "Weighted F1"]].round(3).to_string(index=False))

    print("\n" + "=" * 80)
    print("TASK 2: SEMIOTIC SUBSETS ACCURACY / MICRO F1 [Table VI]")
    print("=" * 80)
    print(df_sub.round(2).to_string(index=False))
    return df_base, df_sub


if __name__ == "__main__":
    main()

