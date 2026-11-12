"""
Task 1: Textual Emotion Detection (FEEL-IT / CamemBERT)
Implements:
  1. Step 1 Base (Raw): EMit A (80%) + MultiEMiT me_train (Raw text) -> emotion_textual_no_corr
  2. Step 2 Raw (Ablation): Initialized from Step 1 -> me_train + mm_train (Raw text) -> emotion_textual_step2_raw
  3. Step 2 Oracle (Train: no rel): Initialized from Step 1 -> me_train + mm_train (Oracle w/o rel) -> emotion_textual_corr
  4. Step 2 Oracle (Train: +rel): Initialized from Step 1 -> me_train + mm_train (Oracle w/ +rel on mm_train) -> emotion_textual_corr_with_rel
Evaluates on the exact held-out MultiEMiT textual test split (me_test, N=405, SPLIT_SEED=42)
and exports summary metrics, per-label classification_reports, 2x2 confusion matrices, and 10x10 co-occurrence matrices.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

from common_utils import (
    EMOTION_LABELS,
    RELATION_LABELS,
    SPLIT_SEED,
    EMOTION_THRESHOLD,
    MAX_LENGTH,
    setup_console,
    seed_everything,
    get_paths,
    oracle_prompt,
    predict_with_threshold,
    metric_dict,
    build_classification_report_df,
    build_multilabel_cm_dfs,
    sigmoid,
    resolve_checkpoint,
)

BASE_MODEL = "MilaNLProc/feel-it-italian-emotion"
AUX_COLS = ["Contenuto", "Contenitore", "Ironia", "Sarcasmo", "Offensivita"] + RELATION_LABELS


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
    # EMit-A: 100% used for training (no discarded 20% validation split)
    df_emit_raw = pd.read_csv(paths["emit_a"])
    emit_tr = df_emit_raw[["text"] + EMOTION_LABELS].dropna(subset=["text"]).reset_index(drop=True)
    for col in EMOTION_LABELS:
        emit_tr[col] = emit_tr[col].astype(int)

    df_me_raw = pd.read_csv(paths["me_textual"])
    for c in AUX_COLS:
        if c not in df_me_raw.columns:
            df_me_raw[c] = 0
    df_me = df_me_raw[["text"] + EMOTION_LABELS + AUX_COLS].dropna(subset=["text"]).fillna(0).reset_index(drop=True)
    me_tr, me_te = train_test_split(df_me, test_size=0.2, random_state=SPLIT_SEED)

    df_mm_tr_raw = pd.read_csv(paths["mm_train"])
    for c in AUX_COLS:
        if c not in df_mm_tr_raw.columns:
            df_mm_tr_raw[c] = 0
    df_mm_tr = df_mm_tr_raw[["text"] + EMOTION_LABELS + AUX_COLS].dropna(subset=["text"]).fillna(0).reset_index(drop=True)

    print(f"Data Partitions: EMit Train={len(emit_tr)}, me_train={len(me_tr)}, mm_train={len(df_mm_tr)}, me_test={len(me_te)}")

    def to_hf_dataset(df_in, text_col="text"):
        df_c = pd.DataFrame({"text": df_in[text_col].astype(str), "labels": df_in[EMOTION_LABELS].values.astype(np.float32).tolist()})
        ds = Dataset.from_pandas(df_c)
        return ds.map(preprocess, batched=True, remove_columns=["text"])

    ds_me_te_raw = to_hf_dataset(me_te, "text")

    # Oracle Prompted me_test (textual dataset has no image/relation)
    me_te_corr = me_te.copy()
    me_te_corr["text_corr"] = oracle_prompt(me_te_corr, "emotion", with_relation=False)
    ds_me_te_corr = to_hf_dataset(me_te_corr, "text_corr")

    df_comb_base = pd.concat([me_tr, df_mm_tr], ignore_index=True)

    stages = [
        {
            "name": "Step 1 Base (Raw)",
            "dir_name": "emotion_textual_no_corr",
            "init_from": BASE_MODEL,
            "train_df": pd.concat([emit_tr[["text"] + EMOTION_LABELS], me_tr[["text"] + EMOTION_LABELS]], ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True),
            "text_col": "text",
            "eval_ds": ds_me_te_raw,
            "prompt": "Raw",
        },
        {
            "name": "Step 2 Raw (Ablation)",
            "dir_name": "emotion_textual_step2_raw",
            "init_from": "emotion_textual_no_corr",
            "train_df": pd.concat([me_tr[["text"] + EMOTION_LABELS], df_mm_tr[["text"] + EMOTION_LABELS]], ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True),
            "text_col": "text",
            "eval_ds": ds_me_te_raw,
            "prompt": "Raw",
        },
        {
            "name": "Step 2 Oracle (Train: no rel)",
            "dir_name": "emotion_textual_corr",
            "init_from": "emotion_textual_no_corr",
            "train_df": df_comb_base.assign(text_corr=oracle_prompt(df_comb_base, "emotion", with_relation=False)).sample(frac=1.0, random_state=args.seed).reset_index(drop=True),
            "text_col": "text_corr",
            "eval_ds": ds_me_te_corr,
            "prompt": "Oracle (Train: no rel, Eval: no rel)",
        },
        {
            "name": "Step 2 Oracle (Train: +rel)",
            "dir_name": "emotion_textual_corr_with_rel",
            "init_from": "emotion_textual_no_corr",
            "train_df": df_comb_base.assign(text_corr=oracle_prompt(df_comb_base, "emotion", with_relation=True)).sample(frac=1.0, random_state=args.seed).reset_index(drop=True),
            "text_col": "text_corr",
            "eval_ds": ds_me_te_corr,
            "prompt": "Oracle (Train: +rel, Eval: no rel)",
        },
    ]

    results = []
    report_dfs = []
    cm_2x2_dfs = []
    cooc_10x10_dfs = []

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
        y_true = np.array(st["eval_ds"]["labels"], dtype=int)

        metrics = metric_dict(y_true, preds)
        per_label_f1 = f1_score(y_true, preds, average=None, zero_division=0)
        rec = {
            "Seed": args.seed,
            "Regime": st["name"],
            "Checkpoint": st["dir_name"],
            "Prompt": st["prompt"],
            "N Test": len(y_true),
            "Micro F1": metrics["Micro F1"],
            "Macro F1": metrics["Macro F1"],
            "Weighted F1": metrics["Weighted F1"],
            "Subset Acc.": metrics["Accuracy"],
        }
        for i, emo_lbl in enumerate(EMOTION_LABELS):
            rec[f"{emo_lbl} F1"] = float(per_label_f1[i])
        results.append(rec)

        meta = {
            "Seed": args.seed,
            "Task": "Task 1: Emotion (Textual me_test)",
            "Regime": st["name"],
            "Checkpoint": st["dir_name"],
            "Prompt": st["prompt"],
        }
        report_dfs.append(build_classification_report_df(y_true, preds, EMOTION_LABELS, meta, multilabel=True))
        df_2x2, df_cooc = build_multilabel_cm_dfs(y_true, preds, EMOTION_LABELS, meta)
        cm_2x2_dfs.append(df_2x2)
        cooc_10x10_dfs.append(df_cooc)

    df_res = pd.DataFrame(results)
    df_res.to_csv(res_dir / "train_textual_emotion_summary.csv", index=False)
    pd.concat(report_dfs, ignore_index=True).to_csv(res_dir / "textual_emotion_classification_reports.csv", index=False)
    pd.concat(cm_2x2_dfs, ignore_index=True).to_csv(res_dir / "textual_emotion_confusion_matrices_2x2.csv", index=False)
    pd.concat(cooc_10x10_dfs, ignore_index=True).to_csv(res_dir / "textual_emotion_cooccurrence_matrices_10x10.csv", index=False)

    print("\n" + "=" * 80)
    print("TASK 1: TEXTUAL EMOTION BENCHMARK (me_test, N=405)")
    print("=" * 80)
    print(df_res[["Regime", "Prompt", "Micro F1", "Macro F1", "Weighted F1", "Subset Acc."]].round(4).to_string(index=False))
    return df_res


if __name__ == "__main__":
    main()
