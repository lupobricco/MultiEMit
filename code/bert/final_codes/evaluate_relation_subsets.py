import os
os.environ['TQDM_NOTEBOOK'] = 'false'
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.multioutput import ClassifierChain
from transformers import AutoModel, AutoTokenizer, pipeline, set_seed
import tqdm.auto, tqdm.std
tqdm.auto.tqdm = tqdm.std.tqdm

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Robust discovery of repository root
repo_root = Path.cwd().resolve()
for _ in range(5):
    if (repo_root / "datasets").exists():
        break
    repo_root = repo_root.parent

DATASETS_DIR = repo_root / "datasets" / "corpora"
IMAGES_DIR = DATASETS_DIR / "images"
CHECKPOINTS_DIR = Path("./checkpoints") if Path("./checkpoints").exists() else repo_root / "code" / "bert" / "checkpoints"

MULTIMODAL_TRAIN_CSV = DATASETS_DIR / "correctly_split" / "multimodal_train_seed42.csv"
MULTIMODAL_TEST_CSV = DATASETS_DIR / "correctly_split" / "multimodal_test_seed42.csv"

EMOTION_LABELS = [
    "Amore", "Anticipazione", "Disgusto", "Fiducia", "Gioia",
    "Neutro", "Paura", "Rabbia", "Sorpresa", "Tristezza"
]
IRONY_LABELS = ["None", "Ironia", "Sarcasmo"]


def create_type_column(row):
    """1=Additiva, 2=Parallela, 3=Divergente"""
    if row["Additiva"] == 1:
        return 1
    elif row["Parallela"] == 1:
        return 2
    elif row["Divergente"] == 1:
        return 3
    else:
        return 0


def row_to_irony_class(row):
    if int(row.get('Sarcasmo', 0)) == 1:
        return 2
    elif int(row.get('Ironia', 0)) == 1:
        return 1
    return 0


def build_multimodal_corr_text(df_in, emotions=False, target=False, irony=False, hate=False, relations=False):
    emotion_cols = EMOTION_LABELS
    out_texts = []
    for _, row in df_in.iterrows():
        em, tg, ih, rel = [], [], [], []
        if emotions:
            for c in emotion_cols:
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
        text = "il seguente testo " + text
        out_texts.append(text)
    return out_texts


def extract_text_cls_features(texts: list, model_path: str, max_length: int = 128, batch_size: int = 32) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    txt_model = AutoModel.from_pretrained(model_path).to(device)
    txt_model.eval()
    all_cls = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = [str(t) for t in texts[start : start + batch_size]]
            tokenized = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_token_type_ids=True,
                return_tensors="pt",
            ).to(device)
            outputs = txt_model(
                input_ids=tokenized.input_ids,
                attention_mask=tokenized.attention_mask,
                output_hidden_states=True,
            )
            cls_emb = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
            all_cls.append(cls_emb)
    return np.concatenate(all_cls, axis=0)


def load_rgb_images(img_paths: pd.Series, images_dir: Path) -> list:
    images = []
    for rel_path in img_paths:
        full_path = images_dir / str(rel_path)
        img = Image.open(full_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        images.append(img)
    return images


def extract_vit_cls_features(images: list, pipe_images, batch_size: int = 32) -> np.ndarray:
    all_img_cls = []
    for start in range(0, len(images), batch_size):
        batch_imgs = images[start : start + batch_size]
        imgs_out = pipe_images(batch_imgs, return_tensors=True, framework="pt")
        batch_cls = np.stack([t[0, 0, :].cpu().numpy() for t in imgs_out], axis=0)
        all_img_cls.append(batch_cls)
    return np.concatenate(all_img_cls, axis=0)


def main():
    # 1. Load data
    train_df = pd.read_csv(MULTIMODAL_TRAIN_CSV)
    test_df = pd.read_csv(MULTIMODAL_TEST_CSV)

    train_df["type"] = train_df.apply(create_type_column, axis=1)
    test_df["type"] = test_df.apply(create_type_column, axis=1)

    y_em_train = train_df[EMOTION_LABELS].values.astype(int)
    y_em_test = test_df[EMOTION_LABELS].values.astype(int)

    train_df["irony_label"] = train_df.apply(row_to_irony_class, axis=1)
    test_df["irony_label"] = test_df.apply(row_to_irony_class, axis=1)
    y_ir_train = train_df["irony_label"].values.astype(int)
    y_ir_test = test_df["irony_label"].values.astype(int)

    test_types = test_df["type"].values
    mask_add = (test_types == 1)
    mask_par = (test_types == 2)
    mask_div = (test_types == 3) | (test_types == 0)

    print(f"Test Set counts -> Total: {len(test_df)}, Additive: {mask_add.sum()}, Divergent: {mask_div.sum()}, Parallel: {mask_par.sum()}")

    # 2. Extract ViT features (google/vit-base-patch16-224)
    VIT_MODEL_ID = "google/vit-base-patch16-224"
    print(f"Loading Vision Encoder: {VIT_MODEL_ID} ...")
    pipe_images = pipeline(task="image-feature-extraction", model=VIT_MODEL_ID, device=device)

    train_images = load_rgb_images(train_df["img_path"], IMAGES_DIR)
    test_images = load_rgb_images(test_df["img_path"], IMAGES_DIR)

    h_img_train = extract_vit_cls_features(train_images, pipe_images)
    h_img_test = extract_vit_cls_features(test_images, pipe_images)

    # 3. Build correlation prompt texts for emotion and irony
    train_df_em_corr = train_df.copy()
    test_df_em_corr = test_df.copy()
    train_df_em_corr["text"] = build_multimodal_corr_text(train_df_em_corr, target=True, irony=True, hate=True, relations=True)
    test_df_em_corr["text"] = build_multimodal_corr_text(test_df_em_corr, target=True, irony=True, hate=True, relations=True)

    train_df_ir_corr = train_df.copy()
    test_df_ir_corr = test_df.copy()
    train_df_ir_corr["text"] = build_multimodal_corr_text(train_df_ir_corr, emotions=True, target=True, hate=True, relations=True)
    test_df_ir_corr["text"] = build_multimodal_corr_text(test_df_ir_corr, emotions=True, target=True, hate=True, relations=True)

    def eval_subsets(y_true, y_pred):
        subsets = {
            "Overall (All 129)": np.ones(len(y_true), dtype=bool),
            "Additive (68)": mask_add,
            "Divergent (38)": mask_div,
            "Parallel (23)": mask_par,
        }
        res = {}
        for name, m in subsets.items():
            yt = y_true[m]
            yp = y_pred[m]
            res[name] = {
                "Macro F1": f1_score(yt, yp, average="macro", zero_division=0),
                "Micro F1": f1_score(yt, yp, average="micro", zero_division=0),
                "Weighted F1": f1_score(yt, yp, average="weighted", zero_division=0),
                "Accuracy": accuracy_score(yt, yp),
            }
        return res

    results_emotion = []
    results_irony = []

    # =========================================================================
    # Task 1: Emotion Detection
    # =========================================================================
    em_checkpoints = []
    if (CHECKPOINTS_DIR / "emotion_textual_no_corr").exists() and (CHECKPOINTS_DIR / "emotion_textual_corr").exists():
        em_checkpoints.append(("Emotion-Checkpoints", str(CHECKPOINTS_DIR / "emotion_textual_no_corr"), str(CHECKPOINTS_DIR / "emotion_textual_corr")))
    elif (CHECKPOINTS_DIR / "concat_emotion_no_corr").exists() and (CHECKPOINTS_DIR / "concat_emotion_corr").exists():
        em_checkpoints.append(("Concatenated", str(CHECKPOINTS_DIR / "concat_emotion_no_corr"), str(CHECKPOINTS_DIR / "concat_emotion_corr")))
    elif (CHECKPOINTS_DIR / "seq_emotion_step2_no_corr").exists() and (CHECKPOINTS_DIR / "seq_emotion_step3_corr").exists():
        em_checkpoints.append(("Sequential", str(CHECKPOINTS_DIR / "seq_emotion_step2_no_corr"), str(CHECKPOINTS_DIR / "seq_emotion_step3_corr")))

    for prefix, ckpt_nc, ckpt_wc in em_checkpoints:
        print(f"\n--- Emotion: {prefix} Models ---")
        h_txt_tr_nc = extract_text_cls_features(train_df["text"].tolist(), ckpt_nc)
        h_txt_te_nc = extract_text_cls_features(test_df["text"].tolist(), ckpt_nc)

        clf_txt_nc = ClassifierChain(
            SGDClassifier(loss="squared_hinge", penalty="l2", n_jobs=-1, random_state=SEED, class_weight="balanced"),
            order="random", random_state=SEED
        )
        clf_txt_nc.fit(h_txt_tr_nc, y_em_train)
        y_pred_txt_nc = clf_txt_nc.predict(h_txt_te_nc).astype(int)

        clf_mm_nc = ClassifierChain(
            SGDClassifier(loss="squared_hinge", penalty="l2", n_jobs=-1, random_state=SEED, class_weight="balanced"),
            order="random", random_state=SEED
        )
        clf_mm_nc.fit(np.concatenate([h_txt_tr_nc, h_img_train], axis=1), y_em_train)
        y_pred_mm_nc = clf_mm_nc.predict(np.concatenate([h_txt_te_nc, h_img_test], axis=1)).astype(int)

        h_txt_tr_wc = extract_text_cls_features(train_df_em_corr["text"].tolist(), ckpt_wc)
        h_txt_te_wc = extract_text_cls_features(test_df_em_corr["text"].tolist(), ckpt_wc)

        clf_txt_wc = ClassifierChain(
            SGDClassifier(loss="squared_hinge", penalty="l2", n_jobs=-1, random_state=SEED, class_weight="balanced"),
            order="random", random_state=SEED
        )
        clf_txt_wc.fit(h_txt_tr_wc, y_em_train)
        y_pred_txt_wc = clf_txt_wc.predict(h_txt_te_wc).astype(int)

        clf_mm_wc = ClassifierChain(
            SGDClassifier(loss="squared_hinge", penalty="l2", n_jobs=-1, random_state=SEED, class_weight="balanced"),
            order="random", random_state=SEED
        )
        clf_mm_wc.fit(np.concatenate([h_txt_tr_wc, h_img_train], axis=1), y_em_train)
        y_pred_mm_wc = clf_mm_wc.predict(np.concatenate([h_txt_te_wc, h_img_test], axis=1)).astype(int)

        for regime_name, y_pred in [
            (f"Text-Only ({prefix}) — No Corr", y_pred_txt_nc),
            (f"Multimodal ({prefix} + ViT) — No Corr", y_pred_mm_nc),
            (f"Text-Only ({prefix}) — With Corr", y_pred_txt_wc),
            (f"Multimodal ({prefix} + ViT) — With Corr", y_pred_mm_wc),
        ]:
            sub_metrics = eval_subsets(y_em_test, y_pred)
            for sname, mvals in sub_metrics.items():
                results_emotion.append({
                    "Regime": regime_name,
                    "Subset": sname,
                    "Macro F1": mvals["Macro F1"],
                    "Micro F1": mvals["Micro F1"],
                    "Weighted F1": mvals["Weighted F1"],
                    "Subset Acc": mvals["Accuracy"],
                })

    # =========================================================================
    # Task 2: Irony & Sarcasm Detection
    # =========================================================================
    ir_ckpt_nc = str(CHECKPOINTS_DIR / "irony_textual_no_corr")
    ir_ckpt_wc = str(CHECKPOINTS_DIR / "irony_textual_corr")

    print(f"\n--- Irony: Models ---")
    h_txt_tr_nc = extract_text_cls_features(train_df["text"].tolist(), ir_ckpt_nc)
    h_txt_te_nc = extract_text_cls_features(test_df["text"].tolist(), ir_ckpt_nc)

    clf_txt_nc = SGDClassifier(loss="squared_hinge", penalty="l2", n_jobs=-1, random_state=SEED, class_weight="balanced")
    clf_txt_nc.fit(h_txt_tr_nc, y_ir_train)
    y_pred_txt_nc = clf_txt_nc.predict(h_txt_te_nc).astype(int)

    clf_mm_nc = SGDClassifier(loss="squared_hinge", penalty="l2", n_jobs=-1, random_state=SEED, class_weight="balanced")
    clf_mm_nc.fit(np.concatenate([h_txt_tr_nc, h_img_train], axis=1), y_ir_train)
    y_pred_mm_nc = clf_mm_nc.predict(np.concatenate([h_txt_te_nc, h_img_test], axis=1)).astype(int)

    h_txt_tr_wc = extract_text_cls_features(train_df_ir_corr["text"].tolist(), ir_ckpt_wc)
    h_txt_te_wc = extract_text_cls_features(test_df_ir_corr["text"].tolist(), ir_ckpt_wc)

    clf_txt_wc = SGDClassifier(loss="squared_hinge", penalty="l2", n_jobs=-1, random_state=SEED, class_weight="balanced")
    clf_txt_wc.fit(h_txt_tr_wc, y_ir_train)
    y_pred_txt_wc = clf_txt_wc.predict(h_txt_te_wc).astype(int)

    clf_mm_wc = SGDClassifier(loss="squared_hinge", penalty="l2", n_jobs=-1, random_state=SEED, class_weight="balanced")
    clf_mm_wc.fit(np.concatenate([h_txt_tr_wc, h_img_train], axis=1), y_ir_train)
    y_pred_mm_wc = clf_mm_wc.predict(np.concatenate([h_txt_te_wc, h_img_test], axis=1)).astype(int)

    for regime_name, y_pred in [
        ("Text-Only (UmBERTo) — No Corr", y_pred_txt_nc),
        ("Multimodal (UmBERTo + ViT) — No Corr", y_pred_mm_nc),
        ("Text-Only (UmBERTo) — With Corr", y_pred_txt_wc),
        ("Multimodal (UmBERTo + ViT) — With Corr", y_pred_mm_wc),
    ]:
        sub_metrics = eval_subsets(y_ir_test, y_pred)
        for sname, mvals in sub_metrics.items():
            results_irony.append({
                "Regime": regime_name,
                "Subset": sname,
                "Macro F1": mvals["Macro F1"],
                "Micro F1": mvals["Micro F1"],
                "Weighted F1": mvals["Weighted F1"],
                "Accuracy": mvals["Accuracy"],
            })

    df_res_em = pd.DataFrame(results_emotion)
    df_res_ir = pd.DataFrame(results_irony)

    print("\n" + "=" * 80)
    print("EMOTION DETECTION: RELATION SUBSETS (MACRO F1)")
    print("=" * 80)
    pivot_em = df_res_em.pivot(index="Regime", columns="Subset", values="Macro F1")[["Overall (All 129)", "Additive (68)", "Divergent (38)", "Parallel (23)"]]
    print(pivot_em.round(4).to_string())

    print("\n" + "=" * 80)
    print("IRONY & SARCASM DETECTION: RELATION SUBSETS (MACRO F1)")
    print("=" * 80)
    pivot_ir = df_res_ir.pivot(index="Regime", columns="Subset", values="Macro F1")[["Overall (All 129)", "Additive (68)", "Divergent (38)", "Parallel (23)"]]
    print(pivot_ir.round(4).to_string())


if __name__ == "__main__":
    main()

