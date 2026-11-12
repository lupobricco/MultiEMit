"""
Task 3: Multimodal Semiotic Relation Evaluation (UmBERTo + ViT-224 on mm_test, N=129)
Evaluates both Regime Cases:
  - Case 1: Stage A Raw (No corr) -> Eval: Raw
  - Case 2: Stage B Oracle (With corr) -> Eval: Oracle (relations=False, target excluded)
Across 4 Linear Probe Variants per Case (Blind Text vs Dual-Encoder x raw vs std = 8 Runs).
Exports baselines CSV, ADD/PAR/DIV per-class F1 CSV, per-label classification_reports CSV,
and 3x3 confusion matrices CSV.
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
    RELATION_LABELS,
    setup_console,
    seed_everything,
    get_paths,
    row_to_relation_class,
    oracle_prompt,
    metric_dict,
    build_classification_report_df,
    build_singlelabel_cm_df,
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

    df_tr = pd.read_csv(paths["mm_train"]).fillna(0)
    df_te = pd.read_csv(paths["mm_test"]).fillna(0)
    y_tr = df_tr.apply(row_to_relation_class, axis=1).values.astype(int)
    y_te = df_te.apply(row_to_relation_class, axis=1).values.astype(int)

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
            "ckpt": "relation_textual_no_corr",
            "regime": "Stage A Raw (No corr)",
            "prompt": "None (Raw)",
            "tr_texts": df_tr["text"].values,
            "te_texts": df_te["text"].values,
        },
        {
            "case_id": "Case 2",
            "ckpt": "relation_textual_corr",
            "regime": "Stage B Oracle (With corr)",
            "prompt": "Oracle",
            "tr_texts": oracle_prompt(df_tr, "relation", with_relation=False),
            "te_texts": oracle_prompt(df_te, "relation", with_relation=False),
        },
    ]

    baseline_rows = []
    class_rows = []
    report_dfs = []
    cm_dfs = []

    for reg in regimes:
        ckpt_path = resolve_checkpoint(reg["ckpt"], args.checkpoints_dir)
        if not (ckpt_path / "config.json").exists():
            print(f"\n[SKIP] {reg['case_id']} ({reg['regime']}): checkpoint '{reg['ckpt']}' not found at {ckpt_path}")
            continue

        print(f"\nExtracting text features for {reg['case_id']}: {reg['regime']} from {ckpt_path} ...")
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
                (f"Blind Text (SGD, {scaling})", f"Multimodal test set (without images, {reg['regime']}, {scaling})", t_tr, t_te),
                (f"Dual-Encoder (UmBERTo + ViT-224, {scaling})", f"Multimodal test set with images (Dual-Encoder, {reg['regime']}, {scaling})", de_tr, de_te),
            ]

            for model_name, cfg_name, X_tr_use, X_te_use in variants:
                clf = fit_linear_head(X_tr_use, y_tr, multilabel=False, seed=args.seed)
                pred = clf.predict(X_te_use).astype(int)
                m = metric_dict(y_te, pred)
                f1_classes = f1_score(y_te, pred, labels=[0, 1, 2], average=None, zero_division=0)

                baseline_rows.append({
                    "Seed": args.seed,
                    "Case": reg["case_id"],
                    "Regime": reg["regime"],
                    "Checkpoint": reg["ckpt"],
                    "Features": scaling,
                    "Scaling": scaling,
                    "Prompting": reg["prompt"],
                    "Model": model_name,
                    "Configuration": cfg_name,
                    "Macro F1": m["Macro F1"],
                    "Micro F1 (= Acc.)": m["Accuracy"],
                    "Weighted F1": m["Weighted F1"],
                    "Additiva F1": float(f1_classes[0]),
                    "Divergente F1": float(f1_classes[1]),
                    "Parallela F1": float(f1_classes[2]),
                })

                class_rows.append({
                    "Seed": args.seed,
                    "Case": reg["case_id"],
                    "Regime": reg["regime"],
                    "Checkpoint": reg["ckpt"],
                    "Prompting": reg["prompt"],
                    "Model": model_name,
                    "Scaling": scaling,
                    "Model Setting": f"{model_name} ({reg['regime']})",
                    "ADD (n=68)": float(f1_classes[0]),
                    "PAR (n=23)": float(f1_classes[2]),
                    "DIV (n=38)": float(f1_classes[1]),
                    "Macro F1": m["Macro F1"],
                    "Micro F1 (= Acc.)": m["Accuracy"],
                })

                meta = {
                    "Seed": args.seed,
                    "Task": "Task 3: Relation (Multimodal mm_test)",
                    "Case": reg["case_id"],
                    "Regime": reg["regime"],
                    "Checkpoint": reg["ckpt"],
                    "Prompting": reg["prompt"],
                    "Model": model_name,
                    "Scaling": scaling,
                }
                report_dfs.append(build_classification_report_df(y_te, pred, RELATION_LABELS, meta, multilabel=False))
                cm_dfs.append(build_singlelabel_cm_df(y_te, pred, RELATION_LABELS, meta))

    df_base = pd.DataFrame(baseline_rows)
    df_cls = pd.DataFrame(class_rows)

    df_base.to_csv(res_dir / "multimodal_relation_baselines.csv", index=False)
    df_cls.to_csv(res_dir / "multimodal_relation_target_classes.csv", index=False)
    if report_dfs:
        pd.concat(report_dfs, ignore_index=True).to_csv(res_dir / "multimodal_relation_classification_reports.csv", index=False)
    if cm_dfs:
        pd.concat(cm_dfs, ignore_index=True).to_csv(res_dir / "multimodal_relation_confusion_matrices.csv", index=False)

    print("\n" + "=" * 80)
    print("TASK 3: MULTIMODAL SEMIOTIC RELATION BASELINES (mm_test, N=129)")
    print("=" * 80)
    if not df_base.empty:
        print(df_base[["Case", "Regime", "Model", "Scaling", "Macro F1", "Micro F1 (= Acc.)", "Weighted F1"]].round(4).to_string(index=False))

    print("\n" + "=" * 80)
    print("TASK 3: TARGET CLASSES F1 BREAKDOWN")
    print("=" * 80)
    if not df_cls.empty:
        print(df_cls[["Case", "Regime", "Model", "Scaling", "ADD (n=68)", "PAR (n=23)", "DIV (n=38)", "Macro F1"]].round(4).to_string(index=False))
    return df_base, df_cls


if __name__ == "__main__":
    main()
