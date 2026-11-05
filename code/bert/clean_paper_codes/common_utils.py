"""
Common utilities for the MultiEMiT paper-ready experimental pipeline.
Implements data loading, prompt generation, metric computation, diagnostic reports
(classification_report and confusion matrices), and feature extraction
strictly aligned with the paper synthesis (IEEE Transactions on Affective Computing).
"""

import os
import sys
import random
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

# Standard Label Spaces
EMOTION_LABELS = [
    "Amore", "Anticipazione", "Disgusto", "Fiducia", "Gioia",
    "Neutro", "Paura", "Rabbia", "Sorpresa", "Tristezza",
]
IRONY_LABELS = ["None", "Ironia", "Sarcasmo"]
RELATION_LABELS = ["Additiva", "Divergente", "Parallela"]

SPLIT_SEED = 42
EMOTION_THRESHOLD = 0.40
MAX_LENGTH = 128


def setup_console():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def seed_everything(seed: int = 42):
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
    raise FileNotFoundError("Could not locate the repository root (containing 'datasets/').")


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


def resolve_checkpoint(ckpt_name: str, local_dir: str = None) -> Path:
    """Finds checkpoint directory containing config.json across local, cwd, and fallback paths."""
    this_dir = Path(__file__).resolve().parent
    candidates = []

    if local_dir is not None:
        p = Path(local_dir)
        if p.is_absolute():
            candidates.append(p / ckpt_name)
        else:
            candidates.append((Path.cwd() / p / ckpt_name).resolve())
            candidates.append((this_dir / p / ckpt_name).resolve())

    # Standard clean_paper_codes/checkpoints, checkpoints_official, and cwd/checkpoints
    candidates.append((this_dir / "checkpoints" / ckpt_name).resolve())
    candidates.append((this_dir / "checkpoints_official" / ckpt_name).resolve())
    candidates.append((Path.cwd() / "checkpoints" / ckpt_name).resolve())
    candidates.append((Path.cwd() / "checkpoints_official" / ckpt_name).resolve())

    # Sibling fallbacks relative to script parent directory
    candidates.append((this_dir.parent / "final_codes_py" / "checkpoints" / ckpt_name).resolve())
    candidates.append((this_dir.parent / "final_codes" / "checkpoints" / ckpt_name).resolve())

    for cand in candidates:
        if (cand / "config.json").exists():
            return cand

    # Default fallback path for creation or reporting
    if local_dir is not None:
        p = Path(local_dir)
        return (p if p.is_absolute() else (this_dir / p)).resolve() / ckpt_name
    return (this_dir / "checkpoints" / ckpt_name).resolve()


# ----------------------------------------------------------------------------------------
# Label mappings
# ----------------------------------------------------------------------------------------
def _flag(row, col: str) -> int:
    val = row.get(col, 0)
    if pd.isna(val):
        return 0
    try:
        return int(float(val))
    except Exception:
        return 0


def row_to_irony_class(row) -> int:
    """0=None, 1=Ironia, 2=Sarcasmo (priority Sarcasmo > Ironia > None)."""
    if _flag(row, "Sarcasmo") == 1:
        return 2
    if _flag(row, "Ironia") == 1:
        return 1
    return 0


def row_to_relation_class(row) -> int:
    """0=Additiva, 1=Divergente, 2=Parallela (priority Parallela > Divergente > Additiva)."""
    if _flag(row, "Parallela") == 1:
        return 2
    if _flag(row, "Divergente") == 1:
        return 1
    return 0


def create_type_column(row) -> int:
    """Semiotic subset mask: 1=Additiva (ADD), 2=Parallela (PAR), 3=Divergente (DIV), 0=none."""
    if _flag(row, "Additiva") == 1:
        return 1
    if _flag(row, "Parallela") == 1:
        return 2
    if _flag(row, "Divergente") == 1:
        return 3
    return 0


# ----------------------------------------------------------------------------------------
# Oracle Prompt Generator
# ----------------------------------------------------------------------------------------
def build_corr_text(df_in, emotions=False, target=False, irony=False, hate=False, relations=False):
    """
    Constructs Italian natural language correlation prompt:
    'il seguente testo [esprime <emozioni>] [ha una relazione <rel>] [contiene <ironia/sarcasmo/offensivita>] [riguarda <target>] <testo>'
    """
    out = []
    for _, row in df_in.iterrows():
        em, tg, ih, rel = [], [], [], []
        if emotions:
            for c in EMOTION_LABELS:
                if _flag(row, c) == 1:
                    em.append(c.lower())
        if target:
            if _flag(row, "Contenuto") == 1:
                tg.append("contenuto")
            if _flag(row, "Contenitore") == 1:
                tg.append("contenitore")
        if irony:
            if _flag(row, "Ironia") == 1:
                ih.append("ironia")
            if _flag(row, "Sarcasmo") == 1:
                ih.append("sarcasmo")
        if hate:
            if _flag(row, "Offensivita") == 1:
                ih.append("offensivita")
        if relations:
            if _flag(row, "Additiva") == 1:
                rel.append("additiva")
            if _flag(row, "Divergente") == 1:
                rel.append("divergente")
            if _flag(row, "Parallela") == 1:
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


def oracle_prompt(df, task: str, with_relation: bool = False):
    """Generates oracle prefix for a task, strictly excluding the target layer."""
    if task == "emotion":
        return build_corr_text(df, target=True, irony=True, hate=True, relations=with_relation)
    elif task == "irony":
        return build_corr_text(df, emotions=True, target=True, hate=True, relations=with_relation)
    elif task == "relation":
        return build_corr_text(df, emotions=True, target=True, irony=True, hate=True, relations=False)
    raise ValueError(f"Unknown task: {task}")


# ----------------------------------------------------------------------------------------
# Metrics, Classification Reports, & Confusion Matrices
# ----------------------------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def predict_with_threshold(probs, threshold=EMOTION_THRESHOLD):
    """Multi-label calibrated threshold with dynamic fallback rule."""
    preds = np.zeros(probs.shape, dtype=int)
    for i, row in enumerate(probs):
        above = np.where(row >= threshold)[0]
        if len(above) > 0:
            preds[i, above] = 1
        else:
            order = np.argsort(row)[::-1]
            preds[i, order[0]] = 1
            if row[order[0]] - row[order[1]] <= 0.10:
                preds[i, order[1]] = 1
    return preds


def metric_dict(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    return {
        "Micro F1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "Macro F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Weighted F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
    }


def compute_subset_metrics(y_true, y_pred, types, multilabel: bool) -> dict:
    """Computes Micro F1, Macro F1, and Accuracy across ALL, ADD, PAR, and DIV subsets."""
    from sklearn.metrics import accuracy_score, f1_score
    types = np.asarray(types)
    masks = {
        "ALL": np.ones(len(types), dtype=bool),
        "ADD": types == 1,
        "PAR": types == 2,
        "DIV": (types == 3) | (types == 0),
    }
    res = {}
    for name, m in masks.items():
        if m.sum() == 0:
            continue
        yt, yp = y_true[m], y_pred[m]
        if multilabel:
            cols = np.where(yt.sum(axis=0) > 0)[0]
            macro = float(f1_score(yt[:, cols], yp[:, cols], average="macro", zero_division=0)) if len(cols) else 0.0
        else:
            labels = np.unique(yt)
            macro = float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0))
        res[name] = {
            "N": int(m.sum()),
            "Micro F1": float(f1_score(yt, yp, average="micro", zero_division=0)),
            "Macro F1": macro,
            "Accuracy": float(accuracy_score(yt, yp)),
        }
    return res


def build_classification_report_df(y_true, y_pred, label_names: list, meta: dict, multilabel: bool = False) -> pd.DataFrame:
    """Builds a tidy DataFrame from sklearn.metrics.classification_report for every label + averages."""
    from sklearn.metrics import classification_report
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    if multilabel:
        rep = classification_report(
            y_true_arr,
            y_pred_arr,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )
    else:
        rep = classification_report(
            y_true_arr,
            y_pred_arr,
            labels=list(range(len(label_names))),
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )

    rows = []
    ordered_keys = list(label_names) + [k for k in ["micro avg", "macro avg", "weighted avg", "samples avg", "accuracy"] if k in rep]
    for k in ordered_keys:
        val = rep[k]
        row = dict(meta)
        row["Label"] = k
        if isinstance(val, dict):
            row["Precision"] = float(val.get("precision", 0.0))
            row["Recall"] = float(val.get("recall", 0.0))
            row["F1-Score"] = float(val.get("f1-score", 0.0))
            row["Support"] = int(val.get("support", 0))
        else:
            row["Precision"] = float(val)
            row["Recall"] = float(val)
            row["F1-Score"] = float(val)
            row["Support"] = int(len(y_true_arr))
        rows.append(row)
    return pd.DataFrame(rows)


def build_singlelabel_cm_df(y_true, y_pred, label_names: list, meta: dict) -> pd.DataFrame:
    """Builds a KxK confusion matrix DataFrame for single-label tasks (rows=True, cols=Pred_<label>)."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    rows = []
    for i, true_name in enumerate(label_names):
        row = dict(meta)
        row["True_Label"] = true_name
        for j, pred_name in enumerate(label_names):
            row[f"Pred_{pred_name}"] = int(cm[i, j])
        row["Total_True"] = int(cm[i, :].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def build_multilabel_cm_dfs(y_true, y_pred, label_names: list, meta: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For multi-label classification (e.g., 10-class Emotion):
      1. Returns per-label 2x2 confusion matrix DataFrame (TN, FP, FN, TP, Precision, Recall, F1, Support).
      2. Returns 10x10 Gold-vs-Predicted Emotion Co-occurrence Matrix DataFrame (plus Pred_None).
    """
    from sklearn.metrics import multilabel_confusion_matrix
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    mcm = multilabel_confusion_matrix(y_true_arr, y_pred_arr)

    rows_2x2 = []
    for i, lbl in enumerate(label_names):
        tn, fp, fn, tp = mcm[i].ravel()
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
        row = dict(meta)
        row.update({
            "Label": lbl,
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "Support": int(tp + fn),
        })
        rows_2x2.append(row)

    # 10x10 Gold vs Predicted Co-occurrence Matrix
    cooc_rows = []
    for i, true_lbl in enumerate(label_names):
        active_mask = y_true_arr[:, i] == 1
        row = dict(meta)
        row["Gold_Emotion"] = true_lbl
        row["Gold_Support"] = int(active_mask.sum())
        if active_mask.sum() > 0:
            sub_pred = y_pred_arr[active_mask]
            for j, pred_lbl in enumerate(label_names):
                row[f"Pred_{pred_lbl}"] = int(sub_pred[:, j].sum())
            row["Pred_None"] = int((sub_pred.sum(axis=1) == 0).sum())
        else:
            for pred_lbl in label_names:
                row[f"Pred_{pred_lbl}"] = 0
            row["Pred_None"] = 0
        cooc_rows.append(row)

    return pd.DataFrame(rows_2x2), pd.DataFrame(cooc_rows)


# ----------------------------------------------------------------------------------------
# Feature Extraction & Classification Heads
# ----------------------------------------------------------------------------------------
def extract_text_features(texts, model_path, device, max_length=MAX_LENGTH, batch_size=32):
    """Extracts frozen last-layer [CLS] embeddings (768-d) and classifier logits."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path)).to(device)
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
    """Extracts frozen [CLS] embeddings (768-d) from ViT-Base-224."""
    import torch
    from PIL import Image
    from transformers import pipeline

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


def fit_linear_head(X_train, y_train, multilabel: bool, seed: int = 42):
    """Fits linear SVM (squared_hinge, balanced). Multi-label wrapped in ClassifierChain."""
    from sklearn.linear_model import SGDClassifier
    from sklearn.multioutput import ClassifierChain

    base = SGDClassifier(loss="squared_hinge", penalty="l2", n_jobs=-1, random_state=seed, class_weight="balanced")
    clf = ClassifierChain(base) if multilabel else base
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_train, y_train)
    return clf
