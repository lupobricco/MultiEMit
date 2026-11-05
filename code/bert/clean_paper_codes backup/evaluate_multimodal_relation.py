"""
Task 3: Multimodal Semiotic Relation Evaluation (Table VII & Table IX in Synthesis)
Evaluates Blind Text vs. Dual-Encoder (UmBERTo + ViT-224) on mm_test (N=129)
across unprompted and correlation-prompted configurations, including ADD/PAR/DIV class breakdowns.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from common_utils import (
    RELATION_LABELS,
    setup_console,
    seed_everything,
    get_paths,
    row_to_relation_class,
    oracle_prompt,
    metric_dict,
    resolve_checkpoint,
    extract_text_features,
    extract_vit_features,
    fit_linear_head,
)


def parse_args():
    parser = argparse.ArgumentParser(description="MultiEMiT Multimodal Semiotic Relation Evaluation")
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
    y_tr = df_tr.apply(row_to_relation_class, axis=1).values
    y_te = df_te.apply(row_to_relation_class, axis=1).values

    # Extract ViT Features (Frozen, 768-d)
    print("\nExtracting ViT-224 visual features ...")
    vit_tr = extract_vit_features(df_tr["img_path"].values, paths["images"], args.vit_model, device, args.batch_size)
    vit_te = extract_vit_features(df_te["img_path"].values, paths["images"], args.vit_model, device, args.batch_size)

    # 1. Stage A Raw text
    ckpt_sa = resolve_checkpoint("relation_textual_no_corr", args.checkpoints_dir)
    print(f"\nExtracting Stage A text features from: {ckpt_sa} ...")
    sa_txt_tr, _ = extract_text_features(df_tr["text"].values, ckpt_sa, device, batch_size=args.batch_size)
    sa_txt_te, _ = extract_text_features(df_te["text"].values, ckpt_sa, device, batch_size=args.batch_size)

    # 2. Stage B Oracle text
    ckpt_sb = resolve_checkpoint("relation_textual_corr", args.checkpoints_dir)
    print(f"\nExtracting Stage B Oracle text features from: {ckpt_sb} ...")
    sb_prompt_tr = oracle_prompt(df_tr, "relation", with_relation=False)
    sb_prompt_te = oracle_prompt(df_te, "relation", with_relation=False)
    sb_txt_tr, _ = extract_text_features(sb_prompt_tr, ckpt_sb, device, batch_size=args.batch_size)
    sb_txt_te, _ = extract_text_features(sb_prompt_te, ckpt_sb, device, batch_size=args.batch_size)

    experiments = [
        {
            "regime": "Stage A Raw (No corr)",
            "prompt": "None (Raw)",
            "txt_tr": sa_txt_tr,
            "txt_te": sa_txt_te,
        },
        {
            "regime": "Stage B Oracle (With corr)",
            "prompt": "Oracle",
            "txt_tr": sb_txt_tr,
            "txt_te": sb_txt_te,
        },
    ]

    baseline_rows = []
    class_rows = []

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
            "Features": "raw",
            "Prompting": exp["prompt"],
            "Model": "Blind Text (SGD)",
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
            "Features": "raw",
            "Prompting": exp["prompt"],
            "Model": "Dual-Encoder (BERT + ViT-224)",
            "Macro F1": m_de["Macro F1"],
            "Micro F1 (= Acc.)": m_de["Accuracy"],
            "Weighted F1": m_de["Weighted F1"],
        })

        # Class Breakdown for Table IX
        f1_classes = f1_score(y_te, pred_de, average=None, zero_division=0)
        class_rows.append({
            "Model Setting": f"Dual-Encoder ({exp['regime']})",
            "ADD (n=68)": float(f1_classes[0]),
            "PAR (n=23)": float(f1_classes[2]),
            "DIV (n=38)": float(f1_classes[1]),
            "Macro F1": m_de["Macro F1"],
        })

    df_base = pd.DataFrame(baseline_rows)
    df_cls = pd.DataFrame(class_rows)

    df_base.to_csv(res_dir / "multimodal_relation_baselines.csv", index=False)
    df_cls.to_csv(res_dir / "multimodal_relation_target_classes.csv", index=False)

    print("\n" + "=" * 80)
    print("TASK 3: MULTIMODAL SEMIOTIC RELATION BASELINES (mm_test, N=129) [Table VII]")
    print("=" * 80)
    print(df_base[["Configuration", "Macro F1", "Micro F1 (= Acc.)", "Weighted F1"]].round(3).to_string(index=False))

    print("\n" + "=" * 80)
    print("TASK 3: TARGET CLASSES F1 BREAKDOWN [Table IX]")
    print("=" * 80)
    print(df_cls.round(3).to_string(index=False))
    return df_base, df_cls


if __name__ == "__main__":
    main()

