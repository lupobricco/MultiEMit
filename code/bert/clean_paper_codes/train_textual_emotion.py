"""
Task 1: Textual Emotion Detection (FEEL-IT / CamemBERT)
Implements:
  1. Step 1 Base: EMit A (80%) + MultiEMiT me_train (Raw text)
  2. Step 2 Raw Ablation: Initialized from Step 1 -> me_train + mm_train (Raw text)
  3. Step 2 Oracle: Initialized from Step 1 -> me_train + mm_train (Oracle prompt)
Evaluates on the exact held-out MultiEMiT textual test split (me_test, N=405, SEED=42).
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset, DatasetDict

from common_utils import (
    EMOTION_LABELS,
    SPLIT_SEED,
    EMOTION_THRESHOLD,
    MAX_LENGTH,
    setup_console,
    seed_everything,
    get_paths,
    oracle_prompt,
    predict_with_threshold,
    metric_dict,
    sigmoid,
    resolve_checkpoint,
)

BASE_MODEL = "MilaNLProc/feel-it-italian-emotion"


def parse_args():
    parser = argparse.ArgumentParser(description="MultiEMiT Textual Emotion Training")
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
    probs = sigmoid(logits)
    preds = predict_with_threshold(probs, threshold=EMOTION_THRESHOLD)
    metrics = metric_dict(labels, preds)
    return {
        "f1_macro": metrics["Macro F1"],
        "f1_micro": metrics["Micro F1"],
        "f1_weighted": metrics["Weighted F1"],
        "accuracy": metrics["Accuracy"],
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
        enc["labels"] = examples["labels"]
        return enc

    # 1. Load Data
    df_emit_raw = pd.read_csv(paths["emit_a"])
    df_emit = df_emit_raw[["text"] + EMOTION_LABELS].dropna(subset=["text"]).reset_index(drop=True)
    emit_tr, _ = train_test_split(df_emit, test_size=0.2, random_state=SPLIT_SEED)

    df_me_raw = pd.read_csv(paths["me_textual"])
    df_me = df_me_raw[["text"] + EMOTION_LABELS + ["Contenuto", "Contenitore", "Ironia", "Sarcasmo", "Offensivita"]].dropna(subset=["text"]).reset_index(drop=True)
    me_tr, me_te = train_test_split(df_me, test_size=0.2, random_state=SPLIT_SEED)

    df_mm_tr_raw = pd.read_csv(paths["mm_train"])
    df_mm_tr = df_mm_tr_raw[["text"] + EMOTION_LABELS + ["Contenuto", "Contenitore", "Ironia", "Sarcasmo", "Offensivita"]].dropna(subset=["text"]).reset_index(drop=True)

    print(f"Data Partitions: EMit Train={len(emit_tr)}, me_train={len(me_tr)}, mm_train={len(df_mm_tr)}, me_test={len(me_te)}")

    def to_hf_dataset(df_in, text_col="text"):
        df_c = pd.DataFrame({"text": df_in[text_col].astype(str), "labels": df_in[EMOTION_LABELS].values.astype(np.float32).tolist()})
        ds = Dataset.from_pandas(df_c)
        return ds.map(preprocess, batched=True, remove_columns=["text"])

    ds_me_te_raw = to_hf_dataset(me_te, "text")

    # Oracle Prompted me_test
    me_te_corr = me_te.copy()
    me_te_corr["text_corr"] = oracle_prompt(me_te_corr, "emotion", with_relation=False)
    ds_me_te_corr = to_hf_dataset(me_te_corr, "text_corr")

    stages = [
        {
            "name": "Step 1 Base (Raw)",
            "dir_name": "emotion_textual_no_corr",
            "init_from": BASE_MODEL,
            "train_df": pd.concat([emit_tr[["text"] + EMOTION_LABELS], me_tr[["text"] + EMOTION_LABELS]], ignore_index=True),
            "text_col": "text",
            "eval_ds": ds_me_te_raw,
            "prompt": "Raw",
        },
        {
            "name": "Step 2 Raw (Ablation)",
            "dir_name": "emotion_textual_step2_raw",
            "init_from": "emotion_textual_no_corr",
            "train_df": pd.concat([me_tr[["text"] + EMOTION_LABELS], df_mm_tr[["text"] + EMOTION_LABELS]], ignore_index=True),
            "text_col": "text",
            "eval_ds": ds_me_te_raw,
            "prompt": "Raw",
        },
        {
            "name": "Step 2 Oracle (With Corr)",
            "dir_name": "emotion_textual_corr",
            "init_from": "emotion_textual_no_corr",
            "train_df": (
                lambda: (
                    df_comb := pd.concat([me_tr, df_mm_tr], ignore_index=True),
                    df_comb.assign(text_corr=oracle_prompt(df_comb, "emotion", with_relation=False))
                )[1]
            )(),
            "text_col": "text_corr",
            "eval_ds": ds_me_te_corr,
            "prompt": "Oracle (no rel)",
        },
    ]

    results = []

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
            print(f"\nTraining {st['name']} -> {out_p} ...")
            init_path = st["init_from"]
            if init_path != BASE_MODEL:
                init_path = str(resolve_checkpoint(init_path, str(ckpt_dir)))
            model = AutoModelForSequenceClassification.from_pretrained(
                init_path,
                problem_type="multi_label_classification",
                num_labels=len(EMOTION_LABELS),
                id2label={i: l for i, l in enumerate(EMOTION_LABELS)},
                label2id={l: i for i, l in enumerate(EMOTION_LABELS)},
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
        probs = sigmoid(preds_out.predictions)
        preds = predict_with_threshold(probs, threshold=EMOTION_THRESHOLD)
        y_true = np.array(st["eval_ds"]["labels"])

        metrics = metric_dict(y_true, preds)
        rec = {
            "Regime": st["name"],
            "Prompt": st["prompt"],
            "N Test": len(y_true),
            "Micro F1": metrics["Micro F1"],
            "Macro F1": metrics["Macro F1"],
            "Weighted F1": metrics["Weighted F1"],
            "Subset Acc.": metrics["Accuracy"],
        }
        results.append(rec)

    df_res = pd.DataFrame(results)
    df_res.to_csv(res_dir / "train_textual_emotion_summary.csv", index=False)

    print("\n" + "=" * 80)
    print("TASK 1: TEXTUAL EMOTION BENCHMARK (me_test, N=405)")
    print("=" * 80)
    print(df_res.round(4).to_string(index=False))
    return df_res


if __name__ == "__main__":
    main()

