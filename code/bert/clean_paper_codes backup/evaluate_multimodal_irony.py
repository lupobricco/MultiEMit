"""
Task 2: Multimodal Irony & Sarcasm Evaluation (Table IV & Table VI in Synthesis)
Evaluates Blind Text vs. Dual-Encoder (UmBERTo + ViT-224) on mm_test (N=129)
across unprompted and correlation-prompted configurations under both std and raw scaling,
including Direct Prediction, ADD/PAR/DIV subsets, and per-class F1 metrics.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

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
    y_tr = df_tr.apply(row_to_irony_class, axis=1).values.astype(int)
    y_te = df_te.apply(row_to_irony_class, axis=1).values.astype(int)
    types_te = df_te.apply(create_type_column, axis=1).values

    # Extract ViT Features (Frozen, 768-d)
    print("\nExtracting ViT-224 visual features ...")
    vit_tr_raw = extract_vit_features(df_tr["img_path"].values, paths["images"], args.vit_model, device, args.batch_size)
    vit_te_raw = extract_vit_features(df_te["img_path"].values, paths["images"], args.vit_model, device, args.batch_size)

    # Standardize ViT features with scaler fitted on mm_train
    scaler_img = StandardScaler().fit(vit_tr_raw)
    vit_tr_std = scaler_img.transform(vit_tr_raw)
    vit_te_std = scaler_img.transform(vit_te_raw)

    regimes = [
        {
            "ckpt": "irony_textual_no_corr",
            "label": "Step 1 Base",
            "prompt": "None (Raw)",
            "tr_texts": df_tr["text"].values,
            "te_texts": df_te["text"].values,
            "is_ablation": False,
        },
        {
            "ckpt": "irony_textual_corr",
            "label": "Step 2 Oracle",
            "prompt": "Oracle (no rel)",
            "tr_texts": oracle_prompt(df_tr, "irony", with_relation=False),
            "te_texts": oracle_prompt(df_te, "irony", with_relation=False),
            "is_ablation": False,
        },
        {
            "ckpt": "irony_textual_corr",
            "label": "Step 2 Oracle (+ rel)",
            "prompt": "Oracle (+ rel)",
            "tr_texts": oracle_prompt(df_tr, "irony", with_relation=True),
            "te_texts": oracle_prompt(df_te, "irony", with_relation=True),
            "is_ablation": True,
        },
        {
            "ckpt": "irony_textual_step2_raw",
            "label": "Step 2 Raw",
            "prompt": "None (Raw)",
            "tr_texts": df_tr["text"].values,
            "te_texts": df_te["text"].values,
            "is_ablation": True,
        },
    ]

    baseline_rows = []
    subset_rows = []

    for reg in regimes:
        try:
            ckpt_path = resolve_checkpoint(reg["ckpt"], args.checkpoints_dir)
        except Exception as e:
            print(f"\n[SKIP] Regime '{reg['label']}' checkpoint '{reg['ckpt']}' not found: {e}")
            continue

        print(f"\nExtracting text features for {reg['label']} ({reg['prompt']}) from: {ckpt_path} ...")
        txt_tr_raw, logits_tr = extract_text_features(reg["tr_texts"], ckpt_path, device, batch_size=args.batch_size)
        txt_te_raw, logits_te = extract_text_features(reg["te_texts"], ckpt_path, device, batch_size=args.batch_size)

        # 1. Direct Transformer Prediction (logits argmax on mm_test text directly)
        pred_direct = np.argmax(logits_te, axis=-1)
        m_direct = metric_dict(y_te, pred_direct)
        f1_cls_dir = f1_score(y_te, pred_direct, labels=[0, 1, 2], average=None, zero_division=0)
        baseline_rows.append({
            "Configuration": f"Multimodal test set (Direct Prediction, {reg['label']})",
            "Prompting": reg["prompt"],
            "Model": "Fine-tuned model (direct prediction)",
            "Scaling": "—",
            "Macro F1": m_direct["Macro F1"],
            "Micro F1 (= Acc.)": m_direct["Accuracy"],
            "Weighted F1": m_direct["Weighted F1"],
            "None F1": float(f1_cls_dir[0]),
            "Ironia F1": float(f1_cls_dir[1]),
            "Sarcasmo F1": float(f1_cls_dir[2]),
            "Is Ablation": reg["is_ablation"],
        })

        # Fit StandardScaler on text features from mm_train
        scaler_txt = StandardScaler().fit(txt_tr_raw)
        txt_tr_std = scaler_txt.transform(txt_tr_raw)
        txt_te_std = scaler_txt.transform(txt_te_raw)

        # 2. Linear Head Evaluations under both 'std' (primary) and 'raw' (ablation)
        for scaling in ["std", "raw"]:
            if scaling == "std":
                t_tr, t_te = txt_tr_std, txt_te_std
                v_tr, v_te = vit_tr_std, vit_te_std
                config_suffix = f"{reg['label']} (No corr, std)" if reg["label"] == "Step 1 Base" else (
                    f"{reg['label']} (With corr, std)" if reg["label"] == "Step 2 Oracle" else f"{reg['label']} (std)"
                )
            else:
                t_tr, t_te = txt_tr_raw, txt_te_raw
                v_tr, v_te = vit_tr_raw, vit_te_raw
                config_suffix = f"{reg['label']} (No corr, raw)" if reg["label"] == "Step 1 Base" else (
                    f"{reg['label']} (no rel, raw)" if reg["label"] == "Step 2 Oracle" else f"{reg['label']} (raw)"
                )

            de_tr = np.hstack([t_tr, v_tr])
            de_te = np.hstack([t_te, v_te])

            # A. Blind Text Head
            clf_blind = fit_linear_head(t_tr, y_tr, multilabel=False, seed=args.seed)
            pred_blind = clf_blind.predict(t_te)
            m_blind = metric_dict(y_te, pred_blind)
            f1_cls_blind = f1_score(y_te, pred_blind, labels=[0, 1, 2], average=None, zero_division=0)
            baseline_rows.append({
                "Configuration": f"Multimodal test set (without images, {config_suffix})",
                "Prompting": reg["prompt"],
                "Model": f"Blind Text (SGD, {scaling})",
                "Scaling": scaling,
                "Macro F1": m_blind["Macro F1"],
                "Micro F1 (= Acc.)": m_blind["Accuracy"],
                "Weighted F1": m_blind["Weighted F1"],
                "None F1": float(f1_cls_blind[0]),
                "Ironia F1": float(f1_cls_blind[1]),
                "Sarcasmo F1": float(f1_cls_blind[2]),
                "Is Ablation": reg["is_ablation"] or (scaling == "raw"),
            })

            # B. Dual-Encoder Head
            clf_de = fit_linear_head(de_tr, y_tr, multilabel=False, seed=args.seed)
            pred_de = clf_de.predict(de_te)
            m_de = metric_dict(y_te, pred_de)
            f1_cls_de = f1_score(y_te, pred_de, labels=[0, 1, 2], average=None, zero_division=0)
            baseline_rows.append({
                "Configuration": f"Multimodal test set with images (Dual-Encoder, {config_suffix})",
                "Prompting": reg["prompt"],
                "Model": f"Dual-Encoder (BERT + ViT-224, {scaling})",
                "Scaling": scaling,
                "Macro F1": m_de["Macro F1"],
                "Micro F1 (= Acc.)": m_de["Accuracy"],
                "Weighted F1": m_de["Weighted F1"],
                "None F1": float(f1_cls_de[0]),
                "Ironia F1": float(f1_cls_de[1]),
                "Sarcasmo F1": float(f1_cls_de[2]),
                "Is Ablation": reg["is_ablation"] or (scaling == "raw" and reg["label"] != "Step 2 Oracle (+ rel)"),
            })

            # Semiotic Subsets for Dual-Encoder
            sub_m = compute_subset_metrics(y_te, pred_de, types_te, multilabel=False)
            subset_rows.append({
                "Model Setting": f"Dual-Encoder ({config_suffix})",
                "Prompting": reg["prompt"],
                "Scaling": scaling,
                "ADD (N=68)": sub_m["ADD"]["Accuracy"],
                "PAR (N=23)": sub_m["PAR"]["Accuracy"],
                "DIV (N=38)": sub_m["DIV"]["Accuracy"],
                "Macro F1": m_de["Macro F1"],
                "Is Ablation": reg["is_ablation"] or (scaling == "raw"),
            })

    df_base = pd.DataFrame(baseline_rows)
    df_sub = pd.DataFrame(subset_rows)

    df_base.to_csv(res_dir / "multimodal_irony_baselines.csv", index=False)
    df_sub.to_csv(res_dir / "multimodal_irony_subsets.csv", index=False)

    print("\n" + "=" * 80)
    print("TASK 2: MULTIMODAL IRONY BASELINES (mm_test, N=129) [Table IV]")
    print("=" * 80)
    print(df_base[["Configuration", "Scaling", "Macro F1", "Micro F1 (= Acc.)", "Ironia F1", "Sarcasmo F1"]].round(3).to_string(index=False))

    print("\n" + "=" * 80)
    print("TASK 2: SEMIOTIC SUBSETS ACCURACY / MICRO F1 [Table VI]")
    print("=" * 80)
    print(df_sub[["Model Setting", "Scaling", "ADD (N=68)", "PAR (N=23)", "DIV (N=38)", "Macro F1"]].round(3).to_string(index=False))
    return df_base, df_sub


if __name__ == "__main__":
    main()
