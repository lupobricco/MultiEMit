"""
Task 3: Textual Semiotic Relation Classification (UmBERTo / RoBERTa)
Implements:
  1. Stage A Raw: UmBERTo fine-tuned on mm_train (516 rows, raw text) -> relation_textual_no_corr
  2. Stage B Oracle: UmBERTo fine-tuned on mm_train (516 rows, Oracle prompt) -> relation_textual_corr
Evaluates on the held-out MultiEMiT multimodal test split (mm_test, N=129, SPLIT_SEED=42)
and exports summary metrics, per-label classification_reports, and 3x3 confusion matrices.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from sklearn.metrics import f1_score

from common_utils import (
    RELATION_LABELS,
    MAX_LENGTH,
    setup_console,
    seed_everything,
    get_paths,
    row_to_relation_class,
    oracle_prompt,
    metric_dict,
    build_classification_report_df,
    build_singlelabel_cm_df,
    resolve_checkpoint,
)

BASE_MODEL = "Musixmatch/umberto-commoncrawl-cased-v1"


def parse_args():
    parser = argparse.ArgumentParser(description="MultiEMiT Textual Semiotic Relation Training")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-retrain", action="store_true", default=False)
    parser.add_argument("--checkpoints-dir", type=str, default="./checkpoints")
    parser.add_argument("--results-dir", type=str, default="./results")
    return parser.parse_args()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    m = metric_dict(labels, preds)
    return {
        "accuracy": m["Accuracy"],
        "f1_macro": m["Macro F1"],
        "f1_micro": m["Micro F1"],
        "f1_weighted": m["Weighted F1"],
    }


def main():
    setup_console()
    args = parse_args()
    seed_everything(args.seed)

    paths = get_paths()
    ckpt_dir = Path(args.checkpoints_dir).resolve()
    res_dir = Path(args.results_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def preprocess(examples):
        enc = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
        enc["labels"] = examples["label"]
        return enc

    # Load Data
    df_tr = pd.read_csv(paths["mm_train"]).fillna(0)
    df_te = pd.read_csv(paths["mm_test"]).fillna(0)
    df_tr["label"] = df_tr.apply(row_to_relation_class, axis=1)
    df_te["label"] = df_te.apply(row_to_relation_class, axis=1)

    print(f"Data Partitions: mm_train={len(df_tr)}, mm_test={len(df_te)}")

    def to_hf_dataset(df_in, text_col="text"):
        df_c = pd.DataFrame({"text": df_in[text_col].astype(str), "label": df_in["label"].values.astype(int)})
        ds = Dataset.from_pandas(df_c)
        return ds.map(preprocess, batched=True, remove_columns=["text"])

    ds_te_raw = to_hf_dataset(df_te, "text")

    df_tr_corr = df_tr.copy()
    df_tr_corr["text_corr"] = oracle_prompt(df_tr_corr, "relation", with_relation=False)
    df_te_corr = df_te.copy()
    df_te_corr["text_corr"] = oracle_prompt(df_te_corr, "relation", with_relation=False)
    ds_te_corr = to_hf_dataset(df_te_corr, "text_corr")

    stages = [
        {
            "name": "Stage A Raw",
            "dir_name": "relation_textual_no_corr",
            "train_df": df_tr.sample(frac=1.0, random_state=args.seed).reset_index(drop=True),
            "text_col": "text",
            "eval_ds": ds_te_raw,
            "prompt": "Raw",
        },
        {
            "name": "Stage B Oracle",
            "dir_name": "relation_textual_corr",
            "train_df": df_tr_corr.sample(frac=1.0, random_state=args.seed).reset_index(drop=True),
            "text_col": "text_corr",
            "eval_ds": ds_te_corr,
            "prompt": "Oracle",
        },
    ]

    results = []
    report_dfs = []
    cm_dfs = []

    for st in stages:
        out_p = ckpt_dir / st["dir_name"]
        existing_ckpt = resolve_checkpoint(st["dir_name"], str(ckpt_dir))
        need_train = args.force_retrain or not (existing_ckpt / "config.json").exists()

        targs = TrainingArguments(
            output_dir=str(out_p),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            weight_decay=args.weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            save_total_limit=1,
            seed=args.seed,
            report_to="none",
        )

        if need_train:
            seed_everything(args.seed)
            print(f"\nTraining {st['name']} -> {out_p} ...")
            model = AutoModelForSequenceClassification.from_pretrained(
                BASE_MODEL,
                problem_type="single_label_classification",
                num_labels=len(RELATION_LABELS),
                id2label={i: l for i, l in enumerate(RELATION_LABELS)},
                label2id={l: i for i, l in enumerate(RELATION_LABELS)},
                ignore_mismatched_sizes=True,
            )
            ds_tr = to_hf_dataset(st["train_df"], st["text_col"])
            trainer = Trainer(
                model=model,
                args=targs,
                train_dataset=ds_tr,
                eval_dataset=st["eval_ds"],
                processing_class=tokenizer,
                compute_metrics=compute_metrics,
            )
            trainer.train()
            trainer.save_model(str(out_p))
            tokenizer.save_pretrained(str(out_p))
            eval_p = out_p
        else:
            print(f"\nRe-using verified checkpoint: {existing_ckpt}")
            eval_p = existing_ckpt
            model = AutoModelForSequenceClassification.from_pretrained(str(eval_p))
            trainer = Trainer(
                model=model,
                args=targs,
                eval_dataset=st["eval_ds"],
                processing_class=tokenizer,
                compute_metrics=compute_metrics,
            )

        preds_out = trainer.predict(st["eval_ds"])
        preds = np.argmax(preds_out.predictions, axis=-1)
        y_true = np.array(st["eval_ds"]["labels"], dtype=int)

        m = metric_dict(y_true, preds)
        per_class_f1 = f1_score(y_true, preds, labels=[0, 1, 2], average=None, zero_division=0)

        rec = {
            "Seed": args.seed,
            "Regime": st["name"],
            "Checkpoint": st["dir_name"],
            "Prompt": st["prompt"],
            "N Test": len(y_true),
            "Micro F1 (= Acc)": m["Accuracy"],
            "Macro F1": m["Macro F1"],
            "Weighted F1": m["Weighted F1"],
            "Additiva F1": float(per_class_f1[0]),
            "Divergente F1": float(per_class_f1[1]),
            "Parallela F1": float(per_class_f1[2]),
        }
        results.append(rec)

        meta = {
            "Seed": args.seed,
            "Task": "Task 3: Relation (Textual mm_test)",
            "Regime": st["name"],
            "Checkpoint": st["dir_name"],
            "Prompt": st["prompt"],
        }
        report_dfs.append(build_classification_report_df(y_true, preds, RELATION_LABELS, meta, multilabel=False))
        cm_dfs.append(build_singlelabel_cm_df(y_true, preds, RELATION_LABELS, meta))

    df_res = pd.DataFrame(results)
    df_res.to_csv(res_dir / "train_textual_relation_summary.csv", index=False)
    pd.concat(report_dfs, ignore_index=True).to_csv(res_dir / "textual_relation_classification_reports.csv", index=False)
    pd.concat(cm_dfs, ignore_index=True).to_csv(res_dir / "textual_relation_confusion_matrices.csv", index=False)

    print("\n" + "=" * 80)
    print("TASK 3: TEXTUAL SEMIOTIC RELATION BENCHMARK (mm_test, N=129)")
    print("=" * 80)
    print(df_res[["Regime", "Prompt", "Micro F1 (= Acc)", "Macro F1", "Additiva F1", "Divergente F1", "Parallela F1"]].round(4).to_string(index=False))
    return df_res


if __name__ == "__main__":
    main()
