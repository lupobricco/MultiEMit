"""
Task 2: Textual Irony & Sarcasm Detection (UmBERTo / RoBERTa)
Implements:
  1. Step 1 Base (Raw): IronITA (80% stratified) + MultiEMiT me_train (Raw text) -> irony_textual_no_corr
  2. Step 2 Raw (Ablation): Initialized from Step 1 -> me_train + mm_train (Raw text) -> irony_textual_step2_raw
  3. Step 2 Oracle (Train: no rel): Initialized from Step 1 -> me_train + mm_train (Oracle w/o rel) -> irony_textual_corr
  4. Step 2 Oracle (Train: +rel): Initialized from Step 1 -> me_train + mm_train (Oracle w/ +rel on mm_train) -> irony_textual_corr_with_rel
Evaluates on the exact held-out MultiEMiT textual test split (me_test, N=405, SPLIT_SEED=42)
and exports summary metrics, per-label classification_reports, and 3x3 confusion matrices.
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
    IRONY_LABELS,
    RELATION_LABELS,
    SPLIT_SEED,
    MAX_LENGTH,
    setup_console,
    seed_everything,
    get_paths,
    row_to_irony_class,
    oracle_prompt,
    metric_dict,
    build_classification_report_df,
    build_singlelabel_cm_df,
    resolve_checkpoint,
)

BASE_MODEL = "Musixmatch/umberto-commoncrawl-cased-v1"


def parse_args():
    parser = argparse.ArgumentParser(description="MultiEMiT Textual Irony Training")
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

    # 1. Load IronITA
    df_ironita_raw = pd.read_excel(paths["ironita"])
    df_ironita_raw["None"] = ((df_ironita_raw["irony"] == 0) & (df_ironita_raw["sarcasm"] == 0)).astype(int)
    df_ironita_raw["Ironia"] = ((df_ironita_raw["irony"] == 1) & (df_ironita_raw["sarcasm"] == 0)).astype(int)
    df_ironita_raw["Sarcasmo"] = (df_ironita_raw["sarcasm"] == 1).astype(int)
    df_ironita_raw["label"] = df_ironita_raw.apply(row_to_irony_class, axis=1)
    df_ironita = df_ironita_raw[["text", "label"]].dropna(subset=["text"]).reset_index(drop=True)
    iron_tr, _ = train_test_split(df_ironita, test_size=0.2, random_state=SPLIT_SEED, stratify=df_ironita["label"])

    # 2. Load MultiEMiT Textual
    df_me_raw = pd.read_csv(paths["me_textual"]).fillna(0)
    for c in RELATION_LABELS:
        if c not in df_me_raw.columns:
            df_me_raw[c] = 0
    df_me_raw["label"] = df_me_raw.apply(row_to_irony_class, axis=1)
    me_tr, me_te = train_test_split(df_me_raw, test_size=0.2, random_state=SPLIT_SEED)

    # 3. Load MultiEMiT Multimodal Train
    df_mm_tr = pd.read_csv(paths["mm_train"]).fillna(0)
    for c in RELATION_LABELS:
        if c not in df_mm_tr.columns:
            df_mm_tr[c] = 0
    df_mm_tr["label"] = df_mm_tr.apply(row_to_irony_class, axis=1)

    print(f"Data Partitions: IronITA Train={len(iron_tr)}, me_train={len(me_tr)}, mm_train={len(df_mm_tr)}, me_test={len(me_te)}")

    def to_hf_dataset(df_in, text_col="text"):
        df_c = pd.DataFrame({"text": df_in[text_col].astype(str), "label": df_in["label"].values.astype(int)})
        ds = Dataset.from_pandas(df_c)
        return ds.map(preprocess, batched=True, remove_columns=["text"])

    ds_me_te_raw = to_hf_dataset(me_te, "text")

    # Oracle Prompted me_test (textual dataset has no image/relation)
    me_te_corr = me_te.copy()
    me_te_corr["text_corr"] = oracle_prompt(me_te_corr, "irony", with_relation=False)
    ds_me_te_corr = to_hf_dataset(me_te_corr, "text_corr")

    df_comb_base = pd.concat([me_tr, df_mm_tr], ignore_index=True).fillna(0)

    stages = [
        {
            "name": "Step 1 Base (Raw)",
            "dir_name": "irony_textual_no_corr",
            "init_from": BASE_MODEL,
            "train_df": pd.concat([iron_tr[["text", "label"]], me_tr[["text", "label"]]], ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True),
            "text_col": "text",
            "eval_ds": ds_me_te_raw,
            "prompt": "Raw",
        },
        {
            "name": "Step 2 Raw (Ablation)",
            "dir_name": "irony_textual_step2_raw",
            "init_from": "irony_textual_no_corr",
            "train_df": pd.concat([me_tr[["text", "label"]], df_mm_tr[["text", "label"]]], ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True),
            "text_col": "text",
            "eval_ds": ds_me_te_raw,
            "prompt": "Raw",
        },
        {
            "name": "Step 2 Oracle (Train: no rel)",
            "dir_name": "irony_textual_corr",
            "init_from": "irony_textual_no_corr",
            "train_df": df_comb_base.assign(text_corr=oracle_prompt(df_comb_base, "irony", with_relation=False)).sample(frac=1.0, random_state=args.seed).reset_index(drop=True),
            "text_col": "text_corr",
            "eval_ds": ds_me_te_corr,
            "prompt": "Oracle (Train: no rel, Eval: no rel)",
        },
        {
            "name": "Step 2 Oracle (Train: +rel)",
            "dir_name": "irony_textual_corr_with_rel",
            "init_from": "irony_textual_no_corr",
            "train_df": df_comb_base.assign(text_corr=oracle_prompt(df_comb_base, "irony", with_relation=True)).sample(frac=1.0, random_state=args.seed).reset_index(drop=True),
            "text_col": "text_corr",
            "eval_ds": ds_me_te_corr,
            "prompt": "Oracle (Train: +rel, Eval: no rel)",
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
            init_path = st["init_from"]
            if init_path != BASE_MODEL:
                init_path = str(resolve_checkpoint(init_path, str(ckpt_dir)))
            model = AutoModelForSequenceClassification.from_pretrained(
                init_path,
                problem_type="single_label_classification",
                num_labels=len(IRONY_LABELS),
                id2label={i: l for i, l in enumerate(IRONY_LABELS)},
                label2id={l: i for i, l in enumerate(IRONY_LABELS)},
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
            "Regime": st["name"],
            "Checkpoint": st["dir_name"],
            "Prompt": st["prompt"],
            "N Test": len(y_true),
            "Micro F1 (= Acc)": m["Accuracy"],
            "Macro F1": m["Macro F1"],
            "Weighted F1": m["Weighted F1"],
            "None F1": float(per_class_f1[0]),
            "Ironia F1": float(per_class_f1[1]),
            "Sarcasmo F1": float(per_class_f1[2]),
        }
        results.append(rec)

        meta = {
            "Task": "Task 2: Irony (Textual me_test)",
            "Regime": st["name"],
            "Checkpoint": st["dir_name"],
            "Prompt": st["prompt"],
        }
        report_dfs.append(build_classification_report_df(y_true, preds, IRONY_LABELS, meta, multilabel=False))
        cm_dfs.append(build_singlelabel_cm_df(y_true, preds, IRONY_LABELS, meta))

    df_res = pd.DataFrame(results)
    df_res.to_csv(res_dir / "train_textual_irony_summary.csv", index=False)
    pd.concat(report_dfs, ignore_index=True).to_csv(res_dir / "textual_irony_classification_reports.csv", index=False)
    pd.concat(cm_dfs, ignore_index=True).to_csv(res_dir / "textual_irony_confusion_matrices.csv", index=False)

    print("\n" + "=" * 80)
    print("TASK 2: TEXTUAL IRONY BENCHMARK (me_test, N=405)")
    print("=" * 80)
    print(df_res[["Regime", "Prompt", "Micro F1 (= Acc)", "Macro F1", "Ironia F1", "Sarcasmo F1"]].round(4).to_string(index=False))
    return df_res


if __name__ == "__main__":
    main()
