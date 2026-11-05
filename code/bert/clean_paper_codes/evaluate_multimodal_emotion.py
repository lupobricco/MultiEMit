"""
Task 1: Multimodal Emotion Evaluation (FEEL-IT + ViT-224 on mm_test, N=129)
Evaluates all 6 Regime Cases:
  - Case 1: Step 1 Base (No corr) -> Eval: Raw
  - Case 2: Step 2 Raw (No corr) -> Eval: Raw
  - Case 3: Step 2 Oracle (Train: no rel) -> Eval: Oracle (no rel)
  - Case 4: Step 2 Oracle (Train: no rel) -> Eval: Oracle (+ rel)
  - Case 5: Step 2 Oracle (Train: +rel) -> Eval: Oracle (no rel)
  - Case 6: Step 2 Oracle (Train: +rel) -> Eval: Oracle (+ rel)
Across 4 Linear Probe Variants per Case (Blind Text vs Dual-Encoder x raw vs std = 24 Runs).
Exports baselines CSV, ADD/PAR/DIV subsets CSV, per-label classification_reports CSV,
2x2 confusion matrices CSV, and 10x10 co-occurrence matrices CSV.
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
    EMOTION_LABELS,
    setup_console,
    seed_everything,
    get_paths,
    oracle_prompt,
    create_type_column,
    metric_dict,
    compute_subset_metrics,
    build_classification_report_df,
    build_multilabel_cm_dfs,
    resolve_checkpoint,
    extract_text_features,
    extract_vit_features,
    fit_linear_head,
)


def parse_args():
    parser = argparse.ArgumentParser(description="MultiEMiT Multimodal Emotion Evaluation")
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

    df_tr = pd.read_csv(paths["mm_train"]).fillna(0)
    df_te = pd.read_csv(paths["mm_test"]).fillna(0)
    y_tr = df_tr[EMOTION_LABELS].values.astype(int)
    y_te = df_te[EMOTION_LABELS].values.astype(int)
    types_te = df_te.apply(create_type_column, axis=1).values

    # Extract ViT Features (Frozen, 768-d)
    print("\nExtracting ViT-224 visual features ...")
    vit_tr_raw = extract_vit_features(df_tr["img_path"].values, paths["images"], args.vit_model, device, args.batch_size)
    vit_te_raw = extract_vit_features(df_te["img_path"].values, paths["images"], args.vit_model, device, args.batch_size)

    scaler_img = StandardScaler().fit(vit_tr_raw)
    vit_tr_std = scaler_img.transform(vit_tr_raw)
    vit_te_std = scaler_img.transform(vit_te_raw)

    regimes = [
        {
            "case_id": "Case 1",
            "ckpt": "emotion_textual_no_corr",
            "label": "Step 1 Base (No corr)",
            "train_rel": "None (Raw)",
            "eval_rel": "None (Raw)",
            "prompt": "None (Raw)",
            "tr_texts": df_tr["text"].values,
            "te_texts": df_te["text"].values,
        },
        {
            "case_id": "Case 2",
            "ckpt": "emotion_textual_step2_raw",
            "label": "Step 2 Raw (No corr)",
            "train_rel": "None (Raw)",
            "eval_rel": "None (Raw)",
            "prompt": "None (Raw)",
            "tr_texts": df_tr["text"].values,
            "te_texts": df_te["text"].values,
        },
        {
            "case_id": "Case 3",
            "ckpt": "emotion_textual_corr",
            "label": "Step 2 Oracle (Train: no rel, Eval: no rel)",
            "train_rel": "no rel",
            "eval_rel": "no rel",
            "prompt": "Oracle (no rel)",
            "tr_texts": oracle_prompt(df_tr, "emotion", with_relation=False),
            "te_texts": oracle_prompt(df_te, "emotion", with_relation=False),
        },
        {
            "case_id": "Case 4",
            "ckpt": "emotion_textual_corr",
            "label": "Step 2 Oracle (Train: no rel, Eval: +rel)",
            "train_rel": "no rel",
            "eval_rel": "+rel",
            "prompt": "Oracle (+ rel)",
            "tr_texts": oracle_prompt(df_tr, "emotion", with_relation=True),
            "te_texts": oracle_prompt(df_te, "emotion", with_relation=True),
        },
        {
            "case_id": "Case 5",
            "ckpt": "emotion_textual_corr_with_rel",
            "label": "Step 2 Oracle (Train: +rel, Eval: no rel)",
            "train_rel": "+rel",
            "eval_rel": "no rel",
            "prompt": "Oracle (no rel)",
            "tr_texts": oracle_prompt(df_tr, "emotion", with_relation=False),
            "te_texts": oracle_prompt(df_te, "emotion", with_relation=False),
        },
        {
            "case_id": "Case 6",
            "ckpt": "emotion_textual_corr_with_rel",
            "label": "Step 2 Oracle (Train: +rel, Eval: +rel)",
            "train_rel": "+rel",
            "eval_rel": "+rel",
            "prompt": "Oracle (+ rel)",
            "tr_texts": oracle_prompt(df_tr, "emotion", with_relation=True),
            "te_texts": oracle_prompt(df_te, "emotion", with_relation=True),
        },
    ]

    baseline_rows = []
    subset_rows = []
    report_dfs = []
    cm_2x2_dfs = []
    cooc_10x10_dfs = []

    for reg in regimes:
        ckpt_path = resolve_checkpoint(reg["ckpt"], args.checkpoints_dir)
        if not (ckpt_path / "config.json").exists():
            print(f"\n[SKIP] {reg['case_id']} ({reg['label']}): checkpoint '{reg['ckpt']}' not found at {ckpt_path}")
            continue

        print(f"\nExtracting text features for {reg['case_id']}: {reg['label']} from {ckpt_path} ...")
        txt_tr_raw, _ = extract_text_features(reg["tr_texts"], ckpt_path, device, batch_size=args.batch_size)
        txt_te_raw, _ = extract_text_features(reg["te_texts"], ckpt_path, device, batch_size=args.batch_size)

        scaler_txt = StandardScaler().fit(txt_tr_raw)
        txt_tr_std = scaler_txt.transform(txt_tr_raw)
        txt_te_std = scaler_txt.transform(txt_te_raw)

        for scaling in ["raw", "std"]:
            if scaling == "raw":
                t_tr, t_te = txt_tr_raw, txt_te_raw
                v_tr, v_te = vit_tr_raw, vit_te_raw
            else:
                t_tr, t_te = txt_tr_std, txt_te_std
                v_tr, v_te = vit_tr_std, vit_te_std

            de_tr = np.hstack([t_tr, v_tr])
            de_te = np.hstack([t_te, v_te])

            variants = [
                ("Blind Text (ClassifierChain)", f"Multimodal test set (without images, {reg['label']}, {scaling})", t_tr, t_te),
                ("Dual-Encoder (FEEL-IT + ViT-224)", f"Multimodal test set with images (Dual-Encoder, {reg['label']}, {scaling})", de_tr, de_te),
            ]

            for model_name, cfg_name, X_tr_use, X_te_use in variants:
                clf = fit_linear_head(X_tr_use, y_tr, multilabel=True, seed=args.seed)
                pred = clf.predict(X_te_use).astype(int)
                m = metric_dict(y_te, pred)
                per_lbl_f1 = f1_score(y_te, pred, average=None, zero_division=0)

                row = {
                    "Case": reg["case_id"],
                    "Regime": reg["label"],
                    "Checkpoint": reg["ckpt"],
                    "Train Relation": reg["train_rel"],
                    "Eval Relation": reg["eval_rel"],
                    "Prompting": reg["prompt"],
                    "Model": model_name,
                    "Scaling": scaling,
                    "Configuration": cfg_name,
                    "Macro F1": m["Macro F1"],
                    "Micro F1": m["Micro F1"],
                    "Weighted F1": m["Weighted F1"],
                    "Subset Acc.": m["Accuracy"],
                }
                for i, emo_lbl in enumerate(EMOTION_LABELS):
                    row[f"{emo_lbl} F1"] = float(per_lbl_f1[i])
                baseline_rows.append(row)

                sub_m = compute_subset_metrics(y_te, pred, types_te, multilabel=True)
                subset_rows.append({
                    "Case": reg["case_id"],
                    "Regime": reg["label"],
                    "Checkpoint": reg["ckpt"],
                    "Prompting": reg["prompt"],
                    "Model": model_name,
                    "Scaling": scaling,
                    "Model Setting": f"{model_name} ({reg['label']}, {scaling})",
                    "ADD (N=68)": sub_m["ADD"]["Micro F1"],
                    "PAR (N=23)": sub_m["PAR"]["Micro F1"],
                    "DIV (N=38)": sub_m["DIV"]["Micro F1"],
                    "ADD Macro F1": sub_m["ADD"]["Macro F1"],
                    "PAR Macro F1": sub_m["PAR"]["Macro F1"],
                    "DIV Macro F1": sub_m["DIV"]["Macro F1"],
                    "Macro F1": m["Macro F1"],
                    "Micro F1": m["Micro F1"],
                })

                meta = {
                    "Task": "Task 1: Emotion (Multimodal mm_test)",
                    "Case": reg["case_id"],
                    "Regime": reg["label"],
                    "Checkpoint": reg["ckpt"],
                    "Prompting": reg["prompt"],
                    "Model": model_name,
                    "Scaling": scaling,
                }
                report_dfs.append(build_classification_report_df(y_te, pred, EMOTION_LABELS, meta, multilabel=True))
                df_2x2, df_cooc = build_multilabel_cm_dfs(y_te, pred, EMOTION_LABELS, meta)
                cm_2x2_dfs.append(df_2x2)
                cooc_10x10_dfs.append(df_cooc)

    df_base = pd.DataFrame(baseline_rows)
    df_sub = pd.DataFrame(subset_rows)

    df_base.to_csv(res_dir / "multimodal_emotion_baselines.csv", index=False)
    df_sub.to_csv(res_dir / "multimodal_emotion_subsets.csv", index=False)
    if report_dfs:
        pd.concat(report_dfs, ignore_index=True).to_csv(res_dir / "multimodal_emotion_classification_reports.csv", index=False)
    if cm_2x2_dfs:
        pd.concat(cm_2x2_dfs, ignore_index=True).to_csv(res_dir / "multimodal_emotion_confusion_matrices_2x2.csv", index=False)
    if cooc_10x10_dfs:
        pd.concat(cooc_10x10_dfs, ignore_index=True).to_csv(res_dir / "multimodal_emotion_cooccurrence_matrices_10x10.csv", index=False)

    print("\n" + "=" * 80)
    print("TASK 1: MULTIMODAL EMOTION BASELINES (mm_test, N=129)")
    print("=" * 80)
    if not df_base.empty:
        print(df_base[["Case", "Regime", "Model", "Scaling", "Macro F1", "Micro F1", "Subset Acc."]].round(4).to_string(index=False))

    print("\n" + "=" * 80)
    print("TASK 1: SEMIOTIC SUBSETS MICRO F1")
    print("=" * 80)
    if not df_sub.empty:
        print(df_sub[["Case", "Regime", "Model", "Scaling", "ADD (N=68)", "PAR (N=23)", "DIV (N=38)", "Macro F1"]].round(4).to_string(index=False))
    return df_base, df_sub


if __name__ == "__main__":
    main()
