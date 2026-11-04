"""
Shared helpers for the MultiEMit pure-Python pipeline.

Contents
  * labels, row->class mappings, repo/dataset path discovery
  * oracle correlation-prompt builder (single implementation used by *every* script)
  * metric helpers (+ ADD/PAR/DIV subset metrics) and the unified result-row schema
  * checkpoint fingerprinting (run_config.json) so stale checkpoints are never silently reused
  * generic textual training stage (train-or-load + predict)
  * generic multimodal dual-encoder experiment (used by the 3 multimodal_dual_encoder_*.py scripts)

Test splits are NEVER changed here: the MultiEMit textual split is always
train_test_split(test_size=0.2, random_state=split_seed) with split_seed=42 by default.
"""

import hashlib
import json
import os
import random
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

EMOTION_LABELS = [
    "Amore", "Anticipazione", "Disgusto", "Fiducia", "Gioia",
    "Neutro", "Paura", "Rabbia", "Sorpresa", "Tristezza",
]
IRONY_LABELS = ["None", "Ironia", "Sarcasmo"]
RELATION_LABELS = ["Additiva", "Divergente", "Parallela"]

TASK_NAMES = {
    "emotion": "Emotion Detection (10-Class Multi-Label)",
    "irony": "Irony & Sarcasm Detection (3-Class Single-Label)",
    "relation": "Semiotic Relation Detection (3-Class Single-Label)",
}

SPLIT_SEED_DEFAULT = 42
CODE_VERSION = "py-v2"
EMOTION_THRESHOLD = 0.4
MAX_LENGTH = 128

# Columns of the unified result schema (per-script *_summary.csv share it).
SUMMARY_COLUMNS = [
    "Task", "Stage", "Regime", "Prefix", "Model", "Features", "N Test",
    "Micro F1", "Macro F1", "Weighted F1", "Accuracy", "Notes",
]


# ----------------------------------------------------------------------------------------
# Environment / paths
# ----------------------------------------------------------------------------------------
def setup_console():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def seed_everything(seed: int):
    import torch
    from transformers import set_seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def find_repo_root() -> Path:
    root = Path.cwd().resolve()
    for _ in range(6):
        if (root / "datasets").exists():
            return root
        root = root.parent
    raise FileNotFoundError("Could not locate the repository root (folder containing 'datasets/').")


def get_paths() -> dict:
    corpora = find_repo_root() / "datasets" / "corpora"
    split_dir = corpora / "correctly_split"
    return {
        "corpora": corpora,
        "images": corpora / "images",
        "emit_a": corpora / "emit" / "cleaned_emit_train_A.csv",
        "ironita": corpora / "ironita" / "ironita_full.xlsx",
        "me_textual": split_dir / "textual_only_cleaned.csv",
        "mm_train": split_dir / "multimodal_train_seed42.csv",
        "mm_test": split_dir / "multimodal_test_seed42.csv",
    }


def maybe_subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Debug helper (--debug-n): keeps the pipeline identical but on a tiny subset."""
    if n and n > 0:
        return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------------------
# Label mappings
# ----------------------------------------------------------------------------------------
def row_to_irony_class(row) -> int:
    """0=None, 1=Ironia, 2=Sarcasmo (priority Sarcasmo > Ironia > None)."""
    if int(row.get("Sarcasmo", 0)) == 1:
        return 2
    if int(row.get("Ironia", 0)) == 1:
        return 1
    return 0


def row_to_relation_class(row) -> int:
    """0=Additiva, 1=Divergente, 2=Parallela (priority Parallela > Divergente > Additiva)."""
    if int(row.get("Parallela", 0)) == 1:
        return 2
    if int(row.get("Divergente", 0)) == 1:
        return 1
    return 0


def create_type_column(row) -> int:
    """Semiotic subset mask: 1=Additiva (ADD), 2=Parallela (PAR), 3=Divergente (DIV), 0=none."""
    if row.get("Additiva", 0) == 1:
        return 1
    if row.get("Parallela", 0) == 1:
        return 2
    if row.get("Divergente", 0) == 1:
        return 3
    return 0


# ----------------------------------------------------------------------------------------
# Oracle correlation prompt (gold labels of the *other* annotation layers)
# ----------------------------------------------------------------------------------------
# Which layers are injected for each task (the task's own target layer is always excluded).
PROMPT_LAYERS = {
    "emotion": dict(target=True, irony=True, hate=True),
    "irony": dict(emotions=True, target=True, hate=True),
    "relation": dict(emotions=True, target=True, irony=True, hate=True),
}


def build_corr_text(df_in, emotions=False, target=False, irony=False, hate=False, relations=False):
    """
    Final format (blocks are only present when non-empty):
      il seguente testo [esprime <emozioni>] [ha una relazione <rel>] [contiene <ironia/sarcasmo/offensivita>]
      [riguarda <contenuto/contenitore>] <testo>
    Columns that do not exist for a row (e.g. relations of textual-only posts) are treated as absent.
    """
    out = []
    for _, row in df_in.iterrows():
        em, tg, ih, rel = [], [], [], []
        if emotions:
            for c in EMOTION_LABELS:
                if int(row.get(c, 0)) == 1:
                    em.append(c.lower())
        if target:
            if int(row.get("Contenuto", 0)) == 1:
                tg.append("contenuto")
            if int(row.get("Contenitore", 0)) == 1:
                tg.append("contenitore")
        if irony:
            if int(row.get("Ironia", 0)) == 1:
                ih.append("ironia")
            if int(row.get("Sarcasmo", 0)) == 1:
                ih.append("sarcasmo")
        if hate:
            if int(row.get("Offensivita", 0)) == 1:
                ih.append("offensivita")
        if relations:
            if int(row.get("Additiva", 0)) == 1:
                rel.append("additiva")
            if int(row.get("Divergente", 0)) == 1:
                rel.append("divergente")
            if int(row.get("Parallela", 0)) == 1:
                rel.append("parallela")
        text = str(row["text"])
        if tg:
            text = "riguarda {} ".format(" e ".join(tg)) + text
        if ih:
            text = "contiene {} ".format(" e ".join(ih)) + text
        if rel:
            text = "ha una relazione {} ".format(" e ".join(rel)) + text
        if em:
            text = "esprime {} ".format(" e ".join(em)) + text
        out.append("il seguente testo " + text)
    return out


def oracle_prompt(df, task: str, with_relation: bool):
    """Oracle prompt for `task`; `with_relation` adds the (gold) semiotic relation when the row has one."""
    kwargs = dict(PROMPT_LAYERS[task])
    kwargs["relations"] = bool(with_relation) and task != "relation"
    return build_corr_text(df, **kwargs)


# ----------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def predict_with_threshold(probs, threshold=EMOTION_THRESHOLD):
    """Multi-label decoding: p >= threshold; fallback = arg-max (+ runner-up if within 0.1)."""
    preds = np.zeros(probs.shape, dtype=int)
    for i, row in enumerate(probs):
        above = np.where(row >= threshold)[0]
        if len(above) > 0:
            preds[i, above] = 1
        else:
            order = np.argsort(row)[::-1]
            preds[i, order[0]] = 1
            if row[order[0]] - row[order[1]] <= 0.1:
                preds[i, order[1]] = 1
    return preds


def metric_dict(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "Micro F1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),  # = subset accuracy for multi-label
    }


def make_row(task, stage, regime, prefix, model, features, y_true, y_pred, notes="") -> dict:
    row = {
        "Task": TASK_NAMES[task],
        "Stage": stage,
        "Regime": regime,
        "Prefix": prefix,
        "Model": model,
        "Features": features,
        "N Test": int(len(y_true)),
    }
    row.update(metric_dict(y_true, y_pred))
    row["Notes"] = notes
    return row


def subset_rows(base: dict, y_true, y_pred, types, multilabel: bool) -> list:
    """
    Metrics on ALL / ADD / PAR / DIV subsets of the multimodal test set.
    For the subsets, macro-F1 is averaged only over classes that have support in that subset.
    """
    from sklearn.metrics import accuracy_score, f1_score

    types = np.asarray(types)
    masks = {
        "ALL": np.ones(len(types), dtype=bool),
        "ADD": types == 1,
        "PAR": types == 2,
        "DIV": (types == 3) | (types == 0),
    }
    out = []
    for name, m in masks.items():
        if m.sum() == 0:
            continue
        yt, yp = y_true[m], y_pred[m]
        if multilabel:
            cols = np.where(yt.sum(axis=0) > 0)[0]
            macro = f1_score(yt[:, cols], yp[:, cols], average="macro", zero_division=0) if len(cols) else 0.0
        else:
            labels = np.unique(yt)
            macro = f1_score(yt, yp, labels=labels, average="macro", zero_division=0)
        rec = {k: base[k] for k in ["Task", "Stage", "Regime", "Prefix", "Model", "Features"]}
        rec.update({
            "Subset": name,
            "N": int(m.sum()),
            "Micro F1": f1_score(yt, yp, average="micro", zero_division=0),
            "Macro F1 (classes with support)": macro,
            "Weighted F1": f1_score(yt, yp, average="weighted", zero_division=0),
            "Accuracy": accuracy_score(yt, yp),
        })
        out.append(rec)
    return out


def class_report(y_true, y_pred, labels_names, multilabel: bool) -> str:
    from sklearn.metrics import classification_report

    kwargs = dict(target_names=labels_names, digits=4, zero_division=0)
    if not multilabel:
        kwargs["labels"] = list(range(len(labels_names)))
    return classification_report(y_true, y_pred, **kwargs)


# ----------------------------------------------------------------------------------------
# Checkpoint fingerprinting (D1): never evaluate a checkpoint that does not match the recipe
# ----------------------------------------------------------------------------------------
RECIPES = {
    "emotion": {
        "step1": "EMitA(80%) + me_train | raw text | init=base",
        "step2_raw": "me_train + mm_train | raw text | init=step1",
        "step2_corr": "me_train + mm_train | oracle prefix | init=step1",
    },
    "irony": {
        "step1": "IronITA(80% stratified) + me_train | raw text | init=base",
        "step2_raw": "me_train + mm_train | raw text | init=step1",
        "step2_corr": "me_train + mm_train | oracle prefix | init=step1",
    },
    "relation": {
        "raw": "mm_train | raw text | init=base",
        "corr": "mm_train | oracle prefix | init=base",
    },
}


def digest(sig: dict) -> str:
    return hashlib.sha1(json.dumps(sig, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def textual_signature(args, task: str, stage: str) -> dict:
    sig = {
        "code": CODE_VERSION,
        "task": task,
        "stage": stage,
        "recipe": RECIPES[task][stage],
        "base_model": args.base_model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "split_seed": getattr(args, "split_seed", None),
        "max_length": MAX_LENGTH,
        "debug_n": getattr(args, "debug_n", 0),
    }
    if stage == "step2_corr":
        sig["mm_relation_in_prefix"] = bool(getattr(args, "mm_relation_prefix", False))
    if stage in ("step2_raw", "step2_corr"):
        sig["parent"] = digest(textual_signature(args, task, "step1"))
    return sig


def _legacy_args_match(ckpt_dir: Path, args) -> bool:
    """For --adopt-legacy: compare the hyper-parameters stored by the Trainer with the current args."""
    import torch

    p = ckpt_dir / "training_args.bin"
    if not p.exists():
        for sub in ckpt_dir.glob("checkpoint-*"):
            if (sub / "training_args.bin").exists():
                p = sub / "training_args.bin"
                break
    if not p.exists():
        return False
    try:
        ta = torch.load(p, weights_only=False)
        return (
            abs(ta.learning_rate - args.lr) < 1e-12
            and int(ta.num_train_epochs) == int(args.epochs)
            and int(ta.per_device_train_batch_size) == int(args.batch_size)
            and abs(ta.weight_decay - args.weight_decay) < 1e-12
            and int(ta.seed) == int(args.seed)
        )
    except Exception as exc:  # pragma: no cover
        print(f"  could not read training_args.bin: {exc}")
        return False


def prepare_checkpoint(out_dir: str, signature: dict, force: bool, adopt_legacy: bool, args) -> bool:
    """
    Returns True if the stage must be (re)trained, False if a matching checkpoint can be re-used.
    Stale / unknown checkpoints are deleted and retrained (use --adopt-legacy only for checkpoints that were
    produced by exactly this recipe before fingerprints existed).
    """
    out = Path(out_dir)
    cfg_path = out / "run_config.json"
    exists = (out / "config.json").exists()

    if exists and not force:
        if cfg_path.exists():
            try:
                saved = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                saved = None
            if saved == signature:
                print(f"[checkpoint] {out.name}: matches current recipe -> re-using (evaluation only).")
                return False
            print(f"[checkpoint] {out.name}: recipe/arguments CHANGED since it was trained -> retraining.")
        elif adopt_legacy and _legacy_args_match(out, args):
            cfg_path.write_text(json.dumps(signature, indent=2, sort_keys=True), encoding="utf-8")
            print(f"[checkpoint] {out.name}: legacy checkpoint ADOPTED (hyper-parameters match training_args.bin).")
            return False
        else:
            print(f"[checkpoint] {out.name}: no run_config.json (unknown provenance) -> retraining. "
                  f"(Use --adopt-legacy only if you are sure it was made with this exact recipe.)")
    elif exists and force:
        print(f"[checkpoint] {out.name}: --force-retrain -> retraining.")

    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    return True


# ----------------------------------------------------------------------------------------
# Generic textual training stage
# ----------------------------------------------------------------------------------------
def run_textual_stage(
    *, out_dir, signature, init_from, train_df, test_df, preprocess, tokenizer, args,
    problem_type, id2label, label2id, compute_metrics, data_collator=None,
    force=False,
):
    """Train (or re-use) one fine-tuning stage and return (logits, label_ids, trained_flag)."""
    from datasets import Dataset, DatasetDict
    from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

    ds = DatasetDict({"train": Dataset.from_pandas(train_df), "test": Dataset.from_pandas(test_df)})
    enc = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)
    enc.set_format("torch")

    need_training = prepare_checkpoint(out_dir, signature, force, getattr(args, "adopt_legacy", False), args)

    targs = TrainingArguments(
        output_dir=out_dir,
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

    if not need_training:
        model = AutoModelForSequenceClassification.from_pretrained(out_dir)
        trainer = Trainer(model=model, args=targs, eval_dataset=enc["test"], processing_class=tokenizer,
                          data_collator=data_collator, compute_metrics=compute_metrics)
    else:
        print(f"Initialising from: {init_from}")
        model = AutoModelForSequenceClassification.from_pretrained(
            init_from, problem_type=problem_type, num_labels=len(id2label),
            id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True,
        )
        trainer = Trainer(model=model, args=targs, train_dataset=enc["train"], eval_dataset=enc["test"],
                          processing_class=tokenizer, data_collator=data_collator, compute_metrics=compute_metrics)
        trainer.train()
        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)
        Path(out_dir, "run_config.json").write_text(json.dumps(signature, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Saved checkpoint to: {out_dir}")

    pred = trainer.predict(enc["test"])
    del trainer, model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred.predictions, pred.label_ids, need_training


def add_textual_cli(parser, base_model: str, with_split_seed: bool = True):
    parser.add_argument("--base-model", type=str, default=base_model)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42, help="Training seed (model init / data order).")
    if with_split_seed:
        parser.add_argument("--split-seed", type=int, default=SPLIT_SEED_DEFAULT,
                            help="Seed of the train/test splits. Keep 42 for comparability with other papers/models.")
    parser.add_argument("--force-retrain", action="store_true", default=False)
    parser.add_argument("--adopt-legacy", action="store_true", default=True,
                        help="Adopt checkpoints that have no run_config.json if their stored hyper-parameters match.")
    parser.add_argument("--debug-n", type=int, default=0, help="Smoke-test: subsample every split to N rows.")
    parser.add_argument("--checkpoints-dir", type=str, default="./checkpoints")
    parser.add_argument("--results-dir", type=str, default="./results")
    return parser


def save_outputs(results_dir: Path, stem: str, rows: list, subsets: list, reports: list):
    results_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)[SUMMARY_COLUMNS]
    df.to_csv(results_dir / f"{stem}_summary.csv", index=False)
    if subsets:
        pd.DataFrame(subsets).to_csv(results_dir / f"{stem}_subsets.csv", index=False)
    with open(results_dir / f"{stem}_reports.txt", "w", encoding="utf-8") as f:
        for title, rep in reports:
            f.write(f"=== {title} ===\n{rep}\n\n")
    print("\n" + "=" * 80)
    print(f"{stem.upper()} RESULTS")
    print(df.drop(columns=["Task", "Weighted F1"]).round(4).to_string(index=False))
    print(f"Saved: {results_dir / (stem + '_summary.csv')}")
    return df


# ----------------------------------------------------------------------------------------
# Multimodal dual-encoder machinery
# ----------------------------------------------------------------------------------------
def extract_text_outputs(texts, model_path, device, max_length=MAX_LENGTH, batch_size=32):
    """Returns (last-layer [CLS] features [N,768], classifier logits [N,K]) of a fine-tuned checkpoint."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"Loading local checkpoint: {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    feats, logits = [], []
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            batch = [str(t) for t in texts[s:s + batch_size]]
            t = tok(batch, padding="max_length", truncation=True, max_length=max_length,
                    return_token_type_ids=True, return_tensors="pt").to(device)
            out = model(input_ids=t.input_ids, attention_mask=t.attention_mask, output_hidden_states=True)
            feats.append(out.hidden_states[-1][:, 0, :].cpu())
            logits.append(out.logits.cpu())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(feats).numpy(), torch.cat(logits).numpy()


def extract_vit_features(img_paths, images_dir: Path, vit_model: str, device, batch_size=32):
    """768-d [CLS] token of google/vit-base-patch16-224 (frozen)."""
    import torch
    from PIL import Image
    from transformers import pipeline

    print(f"Loading Vision Encoder: {vit_model} ...")
    pipe = pipeline(task="image-feature-extraction", model=vit_model, device=device)
    images = []
    for rel in img_paths:
        img = Image.open(images_dir / str(rel))
        images.append(img.convert("RGB") if img.mode != "RGB" else img)
    feats = []
    for s in range(0, len(images), batch_size):
        out = pipe(images[s:s + batch_size], return_tensors=True, framework="pt")
        feats.append(torch.stack([t[0, 0, :].cpu() for t in out], dim=0))
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(feats).numpy()


def feature_variants(txt_tr, img_tr, txt_te, img_te):
    """
    {'raw': unscaled concatenation (as in the first run), 'std': each modality standardised with train statistics}
    -> {name: {'blind': (Xtr, Xte), 'dual': (Xtr, Xte)}}
    """
    from sklearn.preprocessing import StandardScaler

    out = {"raw": {"blind": (txt_tr, txt_te), "dual": (np.hstack([txt_tr, img_tr]), np.hstack([txt_te, img_te]))}}
    st, si = StandardScaler().fit(txt_tr), StandardScaler().fit(img_tr)
    t_tr, t_te = st.transform(txt_tr), st.transform(txt_te)
    i_tr, i_te = si.transform(img_tr), si.transform(img_te)
    out["std"] = {"blind": (t_tr, t_te), "dual": (np.hstack([t_tr, i_tr]), np.hstack([t_te, i_te]))}
    return out


def fit_head(X, y, multilabel: bool, seed: int):
    """SGD (squared-hinge, L2, balanced) head; ClassifierChain for multi-label. Returns (model, n_conv_warnings)."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import SGDClassifier
    from sklearn.multioutput import ClassifierChain

    base = SGDClassifier(loss="squared_hinge", penalty="l2", n_jobs=-1, random_state=seed, class_weight="balanced")
    clf = ClassifierChain(base) if multilabel else base
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clf.fit(X, y)
    n_conv = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    return clf, n_conv


def add_multimodal_cli(parser):
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42, help="Seed of the SGD head.")
    parser.add_argument("--checkpoints-dir", type=str, default="./checkpoints")
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--vit-model", type=str, default="google/vit-base-patch16-224")
    return parser


def run_multimodal_experiment(cfg: dict, args):
    """
    cfg keys: task, stem, multilabel, labels, y_fn, direct_decode ('threshold'|'argmax'), regimes
      regime = dict(ckpt, label, prefix in {None,'oracle','oracle_rel'}, prefix_name, direct(bool))
    For every available regime: [fine-tuned direct prediction] + {blind, dual} x {raw, std} frozen-feature heads.
    """
    import torch

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    task, multilabel, labels = cfg["task"], cfg["multilabel"], cfg["labels"]
    seed_everything(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[Multimodal {task}] device={device} | head seed={args.seed}")

    paths = get_paths()
    ckpt_dir, results_dir = Path(args.checkpoints_dir), Path(args.results_dir)
    train_df = pd.read_csv(paths["mm_train"]).reset_index(drop=True)
    test_df = pd.read_csv(paths["mm_test"]).reset_index(drop=True)
    y_train = cfg["y_fn"](train_df)
    y_test = cfg["y_fn"](test_df)
    types_test = test_df.apply(create_type_column, axis=1).values
    print(f"Multimodal split -> train {len(train_df)} | test {len(test_df)}")

    img_tr = extract_vit_features(train_df["img_path"], paths["images"], args.vit_model, device, args.batch_size)
    img_te = extract_vit_features(test_df["img_path"], paths["images"], args.vit_model, device, args.batch_size)
    print(f"ViT features: train {img_tr.shape} | test {img_te.shape}")

    rows, subsets, reports = [], [], []

    def record(reg, model_name, feat_name, y_pred, notes=""):
        row = make_row(task, "Multimodal split", reg["label"], reg["prefix_name"], model_name, feat_name,
                       y_test, y_pred, notes)
        rows.append(row)
        subsets.extend(subset_rows(row, y_test, y_pred, types_test, multilabel))
        title = f"{reg['label']} | prefix={reg['prefix_name']} | {model_name} | features={feat_name}"
        reports.append((title, class_report(y_test, y_pred, labels, multilabel)))

    for reg in cfg["regimes"]:
        ck = ckpt_dir / reg["ckpt"]
        if not (ck / "config.json").exists():
            print(f"\n[SKIP] regime '{reg['label']}': checkpoint {ck} not found (train the textual stage first).")
            continue
        print("\n" + "=" * 80 + f"\nREGIME: {reg['label']} | prefix={reg['prefix_name']}\n" + "=" * 80)

        if reg["prefix"] is None:
            tr_texts, te_texts = train_df["text"].tolist(), test_df["text"].tolist()
        else:
            with_rel = reg["prefix"] == "oracle_rel"
            tr_texts = oracle_prompt(train_df, task, with_rel)
            te_texts = oracle_prompt(test_df, task, with_rel)

        txt_tr, _ = extract_text_outputs(tr_texts, str(ck), device, batch_size=args.batch_size)
        txt_te, logits_te = extract_text_outputs(te_texts, str(ck), device, batch_size=args.batch_size)

        if reg.get("direct") and cfg.get("direct_decode"):
            if cfg["direct_decode"] == "threshold":
                y_pred = predict_with_threshold(sigmoid(logits_te), EMOTION_THRESHOLD)
            else:
                y_pred = np.argmax(logits_te, axis=-1)
            record(reg, "Fine-tuned model (direct prediction)", "-", y_pred)

        variants = feature_variants(txt_tr, img_tr, txt_te, img_te)
        for feat_name, parts in variants.items():
            for key, model_name in (("blind", "Blind text (frozen CLS + head)"), ("dual", "Dual-Encoder (text + image)")):
                Xtr, Xte = parts[key]
                clf, n_conv = fit_head(Xtr, y_train, multilabel, args.seed)
                y_pred = np.asarray(clf.predict(Xte)).astype(int)
                record(reg, model_name, feat_name, y_pred, notes=f"convergence_warnings={n_conv}")

    if not rows:
        raise RuntimeError("No regime could be evaluated: no textual checkpoints were found.")
    return save_outputs(results_dir, cfg["stem"], rows, subsets, reports)

