"""
Textual Irony & Sarcasm Detection (UmBERTo) — Two-Stage Sequential Training with Raw-Text Step 2 Ablation.

Stages (all evaluated on the MultiEMit textual test split me_test = 405 rows, split at seed 42):
  1. Step 1 (No Correlations):
       - Trained on IronITA (80% stratified train = 3,879 rows) + MultiEMit Textual Train (me_train = 1,618 rows) = 5,497 rows (raw text).
       - Saved to ./checkpoints/irony_textual_no_corr.
  2. Step 2 Raw Ablation (No Correlations, A3):
       - Initialized from Step 1 (./checkpoints/irony_textual_no_corr).
       - Trained on me_train (1,618) + MultiEMit Multimodal Train (mm_train = 516) = 2,134 rows (raw text, no correlations).
       - Isolates the effect of adding mm_train and Step 2 fine-tuning from the effect of correlation prompting.
       - Saved to ./checkpoints/irony_textual_step2_raw.
  3. Step 2 Oracle Correlations (With Correlations):
       - Initialized from Step 1 (./checkpoints/irony_textual_no_corr).
       - Trained on me_train (1,618) + mm_train (516) = 2,134 rows with oracle correlation prefixes
         (emotions + target + offensiveness; no semiotic relation field so train/test schema matches me_test).
       - Saved to ./checkpoints/irony_textual_corr.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from common_utils import (
    IRONY_LABELS,
    MAX_LENGTH,
    add_textual_cli,
    class_report,
    get_paths,
    make_row,
    maybe_subsample,
    oracle_prompt,
    row_to_irony_class,
    run_textual_stage,
    save_outputs,
    seed_everything,
    setup_console,
    textual_signature,
)

id2label = {idx: label for idx, label in enumerate(IRONY_LABELS)}
label2id = {label: idx for idx, label in enumerate(IRONY_LABELS)}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train Textual Irony & Sarcasm Detection (UmBERTo)")
    add_textual_cli(parser, base_model="Musixmatch/umberto-commoncrawl-cased-v1", with_split_seed=True)
    return parser.parse_args()


def main():
    setup_console()
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Irony Textual] device={device} | train_seed={args.seed} | split_seed={args.split_seed}")

    paths = get_paths()
    ckpt_dir = Path(args.checkpoints_dir)
    results_dir = Path(args.results_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    out_step1 = str(ckpt_dir / "irony_textual_no_corr")
    out_step2_raw = str(ckpt_dir / "irony_textual_step2_raw")
    out_step2_corr = str(ckpt_dir / "irony_textual_corr")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def preprocess_singlelabel(examples):
        encoding = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
        encoding["labels"] = examples["label"]
        return encoding

    # 1. Load IronITA and split 80/20 stratified (locked to split_seed=42)
    df_ironita_raw = pd.read_excel(paths["ironita"])
    df_ironita_raw["None"] = ((df_ironita_raw["irony"] == 0) & (df_ironita_raw["sarcasm"] == 0)).astype(int)
    df_ironita_raw["Ironia"] = ((df_ironita_raw["irony"] == 1) & (df_ironita_raw["sarcasm"] == 0)).astype(int)
    df_ironita_raw["Sarcasmo"] = (df_ironita_raw["sarcasm"] == 1).astype(int)
    df_ironita_raw["label"] = df_ironita_raw.apply(row_to_irony_class, axis=1)

    df_ironita = df_ironita_raw[["text", "label"]].dropna(subset=["text"]).reset_index(drop=True)
    iron_train_df, iron_val_df = train_test_split(
        df_ironita, test_size=0.2, random_state=args.split_seed, stratify=df_ironita["label"]
    )
    print(f"IronITA 80/20 Split -> Train: {len(iron_train_df)}, Val: {len(iron_val_df)}")

    # 2. Load MultiEMit Textual & Multimodal Train (split locked to split_seed=42)
    df_multiemit = pd.read_csv(paths["me_textual"])
    df_multiemit["label"] = df_multiemit.apply(row_to_irony_class, axis=1)
    df_multiemit["text_corr"] = oracle_prompt(df_multiemit, "irony", with_relation=False)

    df_mm_train = pd.read_csv(paths["mm_train"])
    df_mm_train["label"] = df_mm_train.apply(row_to_irony_class, axis=1)
    df_mm_train["text_corr"] = oracle_prompt(df_mm_train, "irony", with_relation=False)

    me_train, me_test = train_test_split(df_multiemit, test_size=0.2, random_state=args.split_seed)
    print(f"MultiEMit Textual 80/20 Split -> Train: {len(me_train)}, Test: {len(me_test)} | MM Train: {len(df_mm_train)}")

    # Step 1 (No Corr): IronITA Train (3,879) + me_train (1,618) = 5,497 rows
    step1_train = pd.concat(
        [iron_train_df[["text", "label"]], me_train[["text", "label"]]],
        ignore_index=True,
    ).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    test_raw = me_test[["text", "label"]].reset_index(drop=True)

    # Step 2 Raw Ablation (No Corr): me_train (1,618) + mm_train (516) = 2,134 rows (raw text)
    step2_train_raw = pd.concat(
        [me_train[["text", "label"]], df_mm_train[["text", "label"]]],
        ignore_index=True,
    ).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Step 2 Oracle Corr: me_train (1,618) + mm_train (516) = 2,134 rows (oracle prefix)
    me_train_corr = me_train[["text_corr", "label"]].rename(columns={"text_corr": "text"})
    mm_train_corr = df_mm_train[["text_corr", "label"]].rename(columns={"text_corr": "text"})
    step2_train_corr = pd.concat(
        [me_train_corr, mm_train_corr],
        ignore_index=True,
    ).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    test_corr = me_test[["text_corr", "label"]].rename(columns={"text_corr": "text"}).reset_index(drop=True)

    if args.debug_n:
        step1_train = maybe_subsample(step1_train, args.debug_n, args.seed)
        step2_train_raw = maybe_subsample(step2_train_raw, args.debug_n, args.seed)
        step2_train_corr = maybe_subsample(step2_train_corr, args.debug_n, args.seed)
        test_raw = maybe_subsample(test_raw, args.debug_n, args.seed)
        test_corr = maybe_subsample(test_corr, args.debug_n, args.seed)

    print(f"Step 1 Train (raw): {len(step1_train)} | Step 2 Train (raw/corr): {len(step2_train_raw)} | Test: {len(test_raw)}")

    rows, reports = [], []

    # --------------------------------------------------------------------------
    # STEP 1: IronITA + me_train (Raw text, No Correlations)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 1: IronITA + MultiEMit Textual (Raw Text, No Correlations)")
    print("=" * 80)
    seed_everything(args.seed)
    logits_s1, labels_s1, trained_s1 = run_textual_stage(
        out_dir=out_step1,
        signature=textual_signature(args, "irony", "step1"),
        init_from=args.base_model,
        train_df=step1_train,
        test_df=test_raw,
        preprocess=preprocess_singlelabel,
        tokenizer=tokenizer,
        args=args,
        problem_type="single_label_classification",
        id2label=id2label,
        label2id=label2id,
        compute_metrics=compute_metrics,
        force=args.force_retrain,
    )
    y_true_s1 = labels_s1.astype(int)
    y_pred_s1 = np.argmax(logits_s1, axis=-1)
    rep_s1 = class_report(y_true_s1, y_pred_s1, IRONY_LABELS, multilabel=False)
    print(rep_s1)
    rows.append(make_row(
        "irony", "Textual (me_test)", "Step 1: IronITA + ME Textual", "No",
        "UmBERTo (fine-tuned)", "-", y_true_s1, y_pred_s1,
    ))
    reports.append(("Step 1: IronITA + ME Textual (No Correlations)", rep_s1))

    # --------------------------------------------------------------------------
    # STEP 2 RAW ABLATION: Step 1 Base -> me_train + mm_train (Raw text, No Correlations)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 2 RAW ABLATION: Step 1 Base -> ME Textual + MM Train (Raw Text, No Correlations)")
    print("=" * 80)
    seed_everything(args.seed)
    logits_s2_raw, labels_s2_raw, _ = run_textual_stage(
        out_dir=out_step2_raw,
        signature=textual_signature(args, "irony", "step2_raw"),
        init_from=out_step1,
        train_df=step2_train_raw,
        test_df=test_raw,
        preprocess=preprocess_singlelabel,
        tokenizer=tokenizer,
        args=args,
        problem_type="single_label_classification",
        id2label=id2label,
        label2id=label2id,
        compute_metrics=compute_metrics,
        force=args.force_retrain or trained_s1,
    )
    y_true_s2_raw = labels_s2_raw.astype(int)
    y_pred_s2_raw = np.argmax(logits_s2_raw, axis=-1)
    rep_s2_raw = class_report(y_true_s2_raw, y_pred_s2_raw, IRONY_LABELS, multilabel=False)
    print(rep_s2_raw)
    rows.append(make_row(
        "irony", "Textual (me_test)", "Step 2 Raw: Step 1 -> ME + MM Train", "No",
        "UmBERTo (fine-tuned)", "-", y_true_s2_raw, y_pred_s2_raw, notes="A3 ablation",
    ))
    reports.append(("Step 2 Raw Ablation: Step 1 -> ME Textual + MM Train (No Correlations)", rep_s2_raw))

    # --------------------------------------------------------------------------
    # STEP 2 ORACLE CORR: Step 1 Base -> me_train + mm_train (With Oracle Correlations)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 2 ORACLE CORR: Step 1 Base -> ME Textual + MM Train (With Oracle Correlations)")
    print("=" * 80)
    seed_everything(args.seed)
    logits_s2_corr, labels_s2_corr, _ = run_textual_stage(
        out_dir=out_step2_corr,
        signature=textual_signature(args, "irony", "step2_corr"),
        init_from=out_step1,
        train_df=step2_train_corr,
        test_df=test_corr,
        preprocess=preprocess_singlelabel,
        tokenizer=tokenizer,
        args=args,
        problem_type="single_label_classification",
        id2label=id2label,
        label2id=label2id,
        compute_metrics=compute_metrics,
        force=args.force_retrain or trained_s1,
    )
    y_true_s2_corr = labels_s2_corr.astype(int)
    y_pred_s2_corr = np.argmax(logits_s2_corr, axis=-1)
    rep_s2_corr = class_report(y_true_s2_corr, y_pred_s2_corr, IRONY_LABELS, multilabel=False)
    print(rep_s2_corr)
    rows.append(make_row(
        "irony", "Textual (me_test)", "Step 2 Oracle: Step 1 -> ME + MM Train", "Oracle (no rel)",
        "UmBERTo (fine-tuned)", "-", y_true_s2_corr, y_pred_s2_corr,
    ))
    reports.append(("Step 2 Oracle: Step 1 -> ME Textual + MM Train (With Oracle Correlations)", rep_s2_corr))

    save_outputs(results_dir, "train_textual_irony", rows, subsets=[], reports=reports)


if __name__ == "__main__":
    main()
