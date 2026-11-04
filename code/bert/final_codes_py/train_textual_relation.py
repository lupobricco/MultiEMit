"""
Textual Semiotic Relation Classification (UmBERTo) — Trained on MultiEMit Multimodal Train (516 rows),
evaluated on MultiEMit Multimodal Test (129 rows).

Stages:
  1. Stage A (No Correlations):
       - Trained on mm_train (516 rows, raw text) -> evaluated on mm_test (129 rows, raw text).
       - Saved to ./checkpoints/relation_textual_no_corr.
  2. Stage B (With Oracle Correlations):
       - Trained on mm_train (516 rows, oracle prefix: emotions + target + irony/sarcasm + offensiveness) ->
         evaluated on mm_test (129 rows, oracle prefix).
       - Saved to ./checkpoints/relation_textual_corr.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, DataCollatorWithPadding

from common_utils import (
    MAX_LENGTH,
    RELATION_LABELS,
    add_textual_cli,
    class_report,
    create_type_column,
    get_paths,
    make_row,
    maybe_subsample,
    oracle_prompt,
    row_to_relation_class,
    run_textual_stage,
    save_outputs,
    seed_everything,
    setup_console,
    subset_rows,
    textual_signature,
)

id2label = {idx: label for idx, label in enumerate(RELATION_LABELS)}
label2id = {label: idx for idx, label in enumerate(RELATION_LABELS)}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train Textual Semiotic Relation Detection (UmBERTo)")
    add_textual_cli(parser, base_model="Musixmatch/umberto-commoncrawl-cased-v1", with_split_seed=False)
    return parser.parse_args()


def main():
    setup_console()
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Relation Textual] device={device} | train_seed={args.seed}")

    paths = get_paths()
    ckpt_dir = Path(args.checkpoints_dir)
    results_dir = Path(args.results_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    out_no_corr = str(ckpt_dir / "relation_textual_no_corr")
    out_corr = str(ckpt_dir / "relation_textual_corr")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def preprocess_singlelabel(examples):
        encoding = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
        encoding["labels"] = examples["label"]
        return encoding

    df_train = pd.read_csv(paths["mm_train"])
    df_test = pd.read_csv(paths["mm_test"])
    df_train["label"] = df_train.apply(row_to_relation_class, axis=1)
    df_test["label"] = df_test.apply(row_to_relation_class, axis=1)
    types_test = df_test.apply(create_type_column, axis=1).values

    train_raw = df_train[["text", "label"]].reset_index(drop=True)
    test_raw = df_test[["text", "label"]].reset_index(drop=True)

    df_train_corr = df_train.copy()
    df_test_corr = df_test.copy()
    df_train_corr["text"] = oracle_prompt(df_train_corr, "relation", with_relation=False)
    df_test_corr["text"] = oracle_prompt(df_test_corr, "relation", with_relation=False)
    train_corr = df_train_corr[["text", "label"]].reset_index(drop=True)
    test_corr = df_test_corr[["text", "label"]].reset_index(drop=True)

    if args.debug_n:
        train_raw = maybe_subsample(train_raw, args.debug_n, args.seed)
        test_raw = maybe_subsample(test_raw, args.debug_n, args.seed)
        train_corr = maybe_subsample(train_corr, args.debug_n, args.seed)
        test_corr = maybe_subsample(test_corr, args.debug_n, args.seed)
        types_test = types_test[:len(test_raw)]

    print(f"Relation Multimodal Split -> Train: {len(train_raw)} | Test: {len(test_raw)}")

    rows, subsets, reports = [], [], []

    # --------------------------------------------------------------------------
    # STAGE A: Without Correlations
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE A: Relation Model Without Correlations (MM Train -> MM Test)")
    print("=" * 80)
    seed_everything(args.seed)
    logits_nc, labels_nc, _ = run_textual_stage(
        out_dir=out_no_corr,
        signature=textual_signature(args, "relation", "raw"),
        init_from=args.base_model,
        train_df=train_raw,
        test_df=test_raw,
        preprocess=preprocess_singlelabel,
        tokenizer=tokenizer,
        args=args,
        problem_type="single_label_classification",
        id2label=id2label,
        label2id=label2id,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        force=args.force_retrain,
    )
    y_true_nc = labels_nc.astype(int)
    y_pred_nc = np.argmax(logits_nc, axis=-1)
    rep_nc = class_report(y_true_nc, y_pred_nc, RELATION_LABELS, multilabel=False)
    print(rep_nc)
    row_nc = make_row(
        "relation", "Textual (mm_test)", "Stage A: MM Train Raw", "No",
        "UmBERTo (fine-tuned)", "-", y_true_nc, y_pred_nc,
    )
    rows.append(row_nc)
    subsets.extend(subset_rows(row_nc, y_true_nc, y_pred_nc, types_test, multilabel=False))
    reports.append(("Stage A: Textual Semiotic Relation (No Correlations)", rep_nc))

    # --------------------------------------------------------------------------
    # STAGE B: With Oracle Correlations
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE B: Relation Model With Oracle Correlations (MM Train -> MM Test)")
    print("=" * 80)
    seed_everything(args.seed)
    logits_wc, labels_wc, _ = run_textual_stage(
        out_dir=out_corr,
        signature=textual_signature(args, "relation", "corr"),
        init_from=args.base_model,
        train_df=train_corr,
        test_df=test_corr,
        preprocess=preprocess_singlelabel,
        tokenizer=tokenizer,
        args=args,
        problem_type="single_label_classification",
        id2label=id2label,
        label2id=label2id,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        force=args.force_retrain,
    )
    y_true_wc = labels_wc.astype(int)
    y_pred_wc = np.argmax(logits_wc, axis=-1)
    rep_wc = class_report(y_true_wc, y_pred_wc, RELATION_LABELS, multilabel=False)
    print(rep_wc)
    row_wc = make_row(
        "relation", "Textual (mm_test)", "Stage B: MM Train Oracle", "Oracle",
        "UmBERTo (fine-tuned)", "-", y_true_wc, y_pred_wc,
    )
    rows.append(row_wc)
    subsets.extend(subset_rows(row_wc, y_true_wc, y_pred_wc, types_test, multilabel=False))
    reports.append(("Stage B: Textual Semiotic Relation (With Oracle Correlations)", rep_wc))

    save_outputs(results_dir, "train_textual_relation", rows, subsets=subsets, reports=reports)


if __name__ == "__main__":
    main()
