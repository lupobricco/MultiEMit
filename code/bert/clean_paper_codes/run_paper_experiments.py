"""
Master Runner for MultiEMiT Clean Paper Experiments.
Executes all textual and multimodal experiments across Task 1 (Emotion), Task 2 (Irony),
and Task 3 (Semiotic Relation) and dynamically generates:
  1. results/ALL_EXPERIMENTS_COMPLETE_REPORT.md  (Exhaustive case-by-case report with all tables, classification_reports, & confusion matrices)
  2. results/PAPER_TABLES.md                     (Streamlined publication tables using unified 'raw' protocol with Step 1 Base & Step 2 Raw)
  3. results/paper_summary_tables.md
  4. results/paper_baselines_summary.csv
  5. results/paper_subsets_summary.csv
  6. results/paper_ablation_matrix.csv
  7. results/paper_tables.xlsx
  8. results/paper_tables.tex
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import subprocess

from common_utils import EMOTION_LABELS, IRONY_LABELS, RELATION_LABELS, setup_console


def parse_args():
    parser = argparse.ArgumentParser(description="MultiEMiT Clean Paper Master Runner")
    parser.add_argument("--task", type=str, default="all", choices=["all", "emotion", "irony", "relation"])
    parser.add_argument("--force-retrain", action="store_true", default=False, help="Force retrain textual models from scratch.")
    parser.add_argument("--report-only", action="store_true", default=False, help="Skip training/eval and only regenerate reports from existing CSVs.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoints-dir", type=str, default="./checkpoints")
    parser.add_argument("--results-dir", type=str, default="./results")
    return parser.parse_args()


def run_script(script_name: str, extra_args: list):
    script_path = str(Path(__file__).resolve().parent / script_name)
    cmd = [sys.executable, script_path] + extra_args
    print(f"\n>>> Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, check=True)
    return res.returncode


def _read_csv_if_exists(path: Path):
    return pd.read_csv(path) if path.exists() else None


def _df_to_md(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df is None or df.empty:
        return "_No data available._"
    try:
        return df.to_markdown(index=False, floatfmt=floatfmt)
    except Exception:
        return df.to_string(index=False)


def find_row(df: pd.DataFrame, col: str, pattern: str):
    if df is None or df.empty or col not in df.columns:
        return None
    matches = df[df[col].astype(str).str.contains(pattern, case=False, na=False, regex=False)]
    if not matches.empty:
        return matches.iloc[0]
    return None


def find_exact_mm_row(df: pd.DataFrame, case_id: str, model_pat: str, scaling: str = "raw"):
    if df is None or df.empty:
        return None
    m = (
        (df["Case"].astype(str) == case_id)
        & df["Model"].astype(str).str.contains(model_pat, case=False, na=False, regex=False)
        & (df["Scaling"].astype(str) == scaling)
    )
    sub = df[m]
    return sub.iloc[0] if not sub.empty else None


# ========================================================================================
# 1. Exhaustive Report Builder: ALL_EXPERIMENTS_COMPLETE_REPORT.md
# ========================================================================================
def generate_all_experiments_complete_report(res_dir: Path):
    lines = [
        "# MultiEMiT: Complete Experimental Report (All Runs, Classification Reports & Confusion Matrices)",
        "",
        "> **Auto-Generated Exhaustive Report**: Covers every textual and multimodal experiment across **Task 1 (Emotion Detection)**, **Task 2 (Irony & Sarcasm Detection)**, and **Task 3 (Semiotic Relation Classification)**.",
        "> **Dimensions Evaluated**: `Step 1 Base` vs. `Step 2 Raw` vs. `Step 2 Oracle (Train: no rel)` vs. `Step 2 Oracle (Train: +rel)`; Multimodal `Eval: no rel` vs. `Eval: +rel`; Feature Scaling `raw` (no std) vs. `std` (`StandardScaler`); `Blind Text` vs. `Dual-Encoder`.",
        "",
        "---",
        "",
    ]

    # ------------------------------------------------------------------------------------
    # TASK 1: EMOTION DETECTION
    # ------------------------------------------------------------------------------------
    lines.extend([
        "## 1. Task 1: Emotion Detection (10-Class Multi-Label, `FEEL-IT` + `ViT-224`)",
        "",
        "- **Text Encoder**: `MilaNLProc/feel-it-italian-emotion` (`FEEL-IT`, 768-d `[CLS]`)",
        "- **Vision Encoder**: `google/vit-base-patch16-224` (`ViT-Base-224`, frozen 768-d `[CLS]`)",
        "- **Textual Head (`me_test`, $N=405$)**: Sequence classification head + `EMOTION_THRESHOLD = 0.40` with dynamic top-2 fallback (`prob[0] - prob[1] <= 0.10`)",
        "- **Multimodal Head (`mm_test`, $N=129$)**: `ClassifierChain(SGDClassifier(loss='squared_hinge', penalty='l2', class_weight='balanced', random_state=42))`",
        "",
    ])

    df_t1_txt = _read_csv_if_exists(res_dir / "train_textual_emotion_summary.csv")
    df_t1_txt_rep = _read_csv_if_exists(res_dir / "textual_emotion_classification_reports.csv")
    df_t1_txt_cm2 = _read_csv_if_exists(res_dir / "textual_emotion_confusion_matrices_2x2.csv")
    df_t1_txt_cooc = _read_csv_if_exists(res_dir / "textual_emotion_cooccurrence_matrices_10x10.csv")

    if df_t1_txt is not None:
        lines.extend([
            "### 1.1 Textual Emotion Benchmark (`me_test`, $N=405$)",
            "",
            "#### Overall Metrics Summary",
            _df_to_md(df_t1_txt[["Regime", "Checkpoint", "Prompt", "N Test", "Macro F1", "Micro F1", "Weighted F1", "Subset Acc."]]),
            "",
            "#### Per-Emotion F1 Breakdown (`me_test`)",
            _df_to_md(df_t1_txt[["Regime"] + [f"{c} F1" for c in EMOTION_LABELS if f"{c} F1" in df_t1_txt.columns]]),
            "",
        ])
        for _, r in df_t1_txt.iterrows():
            reg_name = r["Regime"]
            lines.append(f"#### Textual Case: `{reg_name}` (`{r['Checkpoint']}`)")
            if df_t1_txt_rep is not None:
                sub_rep = df_t1_txt_rep[df_t1_txt_rep["Regime"] == reg_name][["Label", "Precision", "Recall", "F1-Score", "Support"]]
                lines.extend(["**Classification Report**:", "", _df_to_md(sub_rep), ""])
            if df_t1_txt_cm2 is not None:
                sub_cm = df_t1_txt_cm2[df_t1_txt_cm2["Regime"] == reg_name][["Label", "TN", "FP", "FN", "TP", "Precision", "Recall", "F1-Score", "Support"]]
                lines.extend(["**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:", "", _df_to_md(sub_cm), ""])
            if df_t1_txt_cooc is not None:
                cooc_cols = ["Gold_Emotion", "Gold_Support"] + [f"Pred_{c}" for c in EMOTION_LABELS] + ["Pred_None"]
                sub_cooc = df_t1_txt_cooc[df_t1_txt_cooc["Regime"] == reg_name][cooc_cols]
                lines.extend(["**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:", "", _df_to_md(sub_cooc, floatfmt=".0f"), ""])

    df_t1_mm = _read_csv_if_exists(res_dir / "multimodal_emotion_baselines.csv")
    df_t1_sub = _read_csv_if_exists(res_dir / "multimodal_emotion_subsets.csv")
    df_t1_mm_rep = _read_csv_if_exists(res_dir / "multimodal_emotion_classification_reports.csv")
    df_t1_mm_cm2 = _read_csv_if_exists(res_dir / "multimodal_emotion_confusion_matrices_2x2.csv")
    df_t1_mm_cooc = _read_csv_if_exists(res_dir / "multimodal_emotion_cooccurrence_matrices_10x10.csv")

    if df_t1_mm is not None and not df_t1_mm.empty:
        lines.extend([
            "---",
            "",
            "### 1.2 Multimodal Emotion Benchmark (`mm_test`, $N=129$ — All 6 Cases $\\times$ `raw` vs `std`)",
            "",
            "#### Master Comparison Table (All Multimodal Emotion Runs)",
            _df_to_md(df_t1_mm[["Case", "Regime", "Train Relation", "Eval Relation", "Model", "Scaling", "Macro F1", "Micro F1", "Weighted F1", "Subset Acc."]]),
            "",
            "#### Per-Emotion F1 Breakdown (All Multimodal Emotion Runs)",
            _df_to_md(df_t1_mm[["Case", "Regime", "Model", "Scaling"] + [f"{c} F1" for c in EMOTION_LABELS if f"{c} F1" in df_t1_mm.columns]]),
            "",
        ])
        if df_t1_sub is not None and not df_t1_sub.empty:
            lines.extend([
                "#### Semiotic Subsets Breakdown (`ADD`, `PAR`, `DIV` on `mm_test`, $N=129$)",
                _df_to_md(df_t1_sub[["Case", "Regime", "Model", "Scaling", "ADD (N=68)", "PAR (N=23)", "DIV (N=38)", "ADD Macro F1", "PAR Macro F1", "DIV Macro F1", "Macro F1", "Micro F1"]]),
                "",
            ])

        lines.extend(["### 1.3 Multimodal Emotion Detailed Reports by Case (`Case 1` – `Case 6`)", ""])
        for case_id, case_group in df_t1_mm.groupby("Case", sort=False):
            reg_label = case_group.iloc[0]["Regime"]
            ckpt_name = case_group.iloc[0]["Checkpoint"]
            prompt_name = case_group.iloc[0]["Prompting"]
            lines.extend([
                f"#### {case_id}: {reg_label} (Checkpoint: `{ckpt_name}`, Eval Prompt: `{prompt_name}`)",
                "",
                _df_to_md(case_group[["Model", "Scaling", "Macro F1", "Micro F1", "Weighted F1", "Subset Acc."]]),
                "",
            ])
            for _, run_row in case_group.iterrows():
                m_name = run_row["Model"]
                sc = run_row["Scaling"]
                lines.append(f"##### Run: `{case_id}` | `{m_name}` | Scaling: `{sc}`")
                if df_t1_mm_rep is not None:
                    mask = (df_t1_mm_rep["Case"] == case_id) & (df_t1_mm_rep["Model"] == m_name) & (df_t1_mm_rep["Scaling"] == sc)
                    lines.extend(["**Classification Report**:", "", _df_to_md(df_t1_mm_rep[mask][["Label", "Precision", "Recall", "F1-Score", "Support"]]), ""])
                if df_t1_mm_cm2 is not None:
                    mask = (df_t1_mm_cm2["Case"] == case_id) & (df_t1_mm_cm2["Model"] == m_name) & (df_t1_mm_cm2["Scaling"] == sc)
                    lines.extend(["**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:", "", _df_to_md(df_t1_mm_cm2[mask][["Label", "TN", "FP", "FN", "TP", "Precision", "Recall", "F1-Score", "Support"]]), ""])
                if df_t1_mm_cooc is not None:
                    mask = (df_t1_mm_cooc["Case"] == case_id) & (df_t1_mm_cooc["Model"] == m_name) & (df_t1_mm_cooc["Scaling"] == sc)
                    cooc_cols = ["Gold_Emotion", "Gold_Support"] + [f"Pred_{c}" for c in EMOTION_LABELS] + ["Pred_None"]
                    lines.extend(["**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:", "", _df_to_md(df_t1_mm_cooc[mask][cooc_cols], floatfmt=".0f"), ""])

    # ------------------------------------------------------------------------------------
    # TASK 2: IRONY & SARCASM DETECTION
    # ------------------------------------------------------------------------------------
    lines.extend([
        "---",
        "",
        "## 2. Task 2: Irony & Sarcasm Detection (3-Class Single-Label, `UmBERTo` + `ViT-224`)",
        "",
        "- **Text Encoder**: `Musixmatch/umberto-commoncrawl-cased-v1` (`UmBERTo`, 768-d `[CLS]`)",
        "- **Vision Encoder**: `google/vit-base-patch16-224` (`ViT-Base-224`, frozen 768-d `[CLS]`)",
        "- **Multimodal Head (`mm_test`, $N=129$)**: `SGDClassifier(loss='squared_hinge', penalty='l2', class_weight='balanced', random_state=42)`",
        "",
    ])

    df_t2_txt = _read_csv_if_exists(res_dir / "train_textual_irony_summary.csv")
    df_t2_txt_rep = _read_csv_if_exists(res_dir / "textual_irony_classification_reports.csv")
    df_t2_txt_cm = _read_csv_if_exists(res_dir / "textual_irony_confusion_matrices.csv")

    if df_t2_txt is not None:
        lines.extend([
            "### 2.1 Textual Irony Benchmark (`me_test`, $N=405$)",
            "",
            _df_to_md(df_t2_txt[["Regime", "Checkpoint", "Prompt", "N Test", "Macro F1", "Micro F1 (= Acc)", "Weighted F1", "None F1", "Ironia F1", "Sarcasmo F1"]]),
            "",
        ])
        for _, r in df_t2_txt.iterrows():
            reg_name = r["Regime"]
            lines.append(f"#### Textual Case: `{reg_name}` (`{r['Checkpoint']}`)")
            if df_t2_txt_rep is not None:
                sub_rep = df_t2_txt_rep[df_t2_txt_rep["Regime"] == reg_name][["Label", "Precision", "Recall", "F1-Score", "Support"]]
                lines.extend(["**Classification Report**:", "", _df_to_md(sub_rep), ""])
            if df_t2_txt_cm is not None:
                cm_cols = ["True_Label"] + [f"Pred_{c}" for c in IRONY_LABELS] + ["Total_True"]
                sub_cm = df_t2_txt_cm[df_t2_txt_cm["Regime"] == reg_name][cm_cols]
                lines.extend(["**3x3 Confusion Matrix**:", "", _df_to_md(sub_cm, floatfmt=".0f"), ""])

    df_t2_mm = _read_csv_if_exists(res_dir / "multimodal_irony_baselines.csv")
    df_t2_sub = _read_csv_if_exists(res_dir / "multimodal_irony_subsets.csv")
    df_t2_mm_rep = _read_csv_if_exists(res_dir / "multimodal_irony_classification_reports.csv")
    df_t2_mm_cm = _read_csv_if_exists(res_dir / "multimodal_irony_confusion_matrices.csv")

    if df_t2_mm is not None and not df_t2_mm.empty:
        lines.extend([
            "---",
            "",
            "### 2.2 Multimodal Irony Benchmark (`mm_test`, $N=129$ — All 6 Cases $\\times$ `raw` vs `std`)",
            "",
            "#### Master Comparison Table (All Multimodal Irony Runs)",
            _df_to_md(df_t2_mm[["Case", "Regime", "Train Relation", "Eval Relation", "Model", "Scaling", "Macro F1", "Micro F1 (= Acc.)", "Weighted F1", "None F1", "Ironia F1", "Sarcasmo F1"]]),
            "",
        ])
        if df_t2_sub is not None and not df_t2_sub.empty:
            lines.extend([
                "#### Semiotic Subsets Breakdown (`ADD`, `PAR`, `DIV` Accuracy / Micro F1 on `mm_test`, $N=129$)",
                _df_to_md(df_t2_sub[["Case", "Regime", "Model", "Scaling", "ADD (N=68)", "PAR (N=23)", "DIV (N=38)", "Macro F1", "Micro F1 (= Acc.)"]]),
                "",
            ])

        lines.extend(["### 2.3 Multimodal Irony Detailed Reports by Case (`Case 1` – `Case 6`)", ""])
        for case_id, case_group in df_t2_mm.groupby("Case", sort=False):
            reg_label = case_group.iloc[0]["Regime"]
            ckpt_name = case_group.iloc[0]["Checkpoint"]
            prompt_name = case_group.iloc[0]["Prompting"]
            lines.extend([
                f"#### {case_id}: {reg_label} (Checkpoint: `{ckpt_name}`, Eval Prompt: `{prompt_name}`)",
                "",
                _df_to_md(case_group[["Model", "Scaling", "Macro F1", "Micro F1 (= Acc.)", "Weighted F1", "None F1", "Ironia F1", "Sarcasmo F1"]]),
                "",
            ])
            for _, run_row in case_group.iterrows():
                m_name = run_row["Model"]
                sc = run_row["Scaling"]
                lines.append(f"##### Run: `{case_id}` | `{m_name}` | Scaling: `{sc}`")
                if df_t2_mm_rep is not None:
                    mask = (df_t2_mm_rep["Case"] == case_id) & (df_t2_mm_rep["Model"] == m_name) & (df_t2_mm_rep["Scaling"] == sc)
                    lines.extend(["**Classification Report**:", "", _df_to_md(df_t2_mm_rep[mask][["Label", "Precision", "Recall", "F1-Score", "Support"]]), ""])
                if df_t2_mm_cm is not None:
                    mask = (df_t2_mm_cm["Case"] == case_id) & (df_t2_mm_cm["Model"] == m_name) & (df_t2_mm_cm["Scaling"] == sc)
                    cm_cols = ["True_Label"] + [f"Pred_{c}" for c in IRONY_LABELS] + ["Total_True"]
                    lines.extend(["**3x3 Confusion Matrix**:", "", _df_to_md(df_t2_mm_cm[mask][cm_cols], floatfmt=".0f"), ""])

    # ------------------------------------------------------------------------------------
    # TASK 3: SEMIOTIC RELATION CLASSIFICATION
    # ------------------------------------------------------------------------------------
    lines.extend([
        "---",
        "",
        "## 3. Task 3: Semiotic Relation Classification (3-Class Single-Label, `UmBERTo` + `ViT-224`)",
        "",
        "- **Text Encoder**: `Musixmatch/umberto-commoncrawl-cased-v1` (`UmBERTo`, 768-d `[CLS]`)",
        "- **Vision Encoder**: `google/vit-base-patch16-224` (`ViT-Base-224`, frozen 768-d `[CLS]`)",
        "- **Multimodal Head (`mm_test`, $N=129$)**: `SGDClassifier(loss='squared_hinge', penalty='l2', class_weight='balanced', random_state=42)`",
        "",
    ])

    df_t3_txt = _read_csv_if_exists(res_dir / "train_textual_relation_summary.csv")
    df_t3_txt_rep = _read_csv_if_exists(res_dir / "textual_relation_classification_reports.csv")
    df_t3_txt_cm = _read_csv_if_exists(res_dir / "textual_relation_confusion_matrices.csv")

    if df_t3_txt is not None:
        lines.extend([
            "### 3.1 Textual Encoder Training Benchmark (`mm_test`, $N=129$)",
            "",
            _df_to_md(df_t3_txt[["Regime", "Checkpoint", "Prompt", "N Test", "Macro F1", "Micro F1 (= Acc)", "Weighted F1", "Additiva F1", "Divergente F1", "Parallela F1"]]),
            "",
        ])
        for _, r in df_t3_txt.iterrows():
            reg_name = r["Regime"]
            lines.append(f"#### Textual Encoder Stage: `{reg_name}` (`{r['Checkpoint']}`)")
            if df_t3_txt_rep is not None:
                sub_rep = df_t3_txt_rep[df_t3_txt_rep["Regime"] == reg_name][["Label", "Precision", "Recall", "F1-Score", "Support"]]
                lines.extend(["**Classification Report**:", "", _df_to_md(sub_rep), ""])
            if df_t3_txt_cm is not None:
                cm_cols = ["True_Label"] + [f"Pred_{c}" for c in RELATION_LABELS] + ["Total_True"]
                sub_cm = df_t3_txt_cm[df_t3_txt_cm["Regime"] == reg_name][cm_cols]
                lines.extend(["**3x3 Confusion Matrix**:", "", _df_to_md(sub_cm, floatfmt=".0f"), ""])

    df_t3_mm = _read_csv_if_exists(res_dir / "multimodal_relation_baselines.csv")
    df_t3_cls = _read_csv_if_exists(res_dir / "multimodal_relation_target_classes.csv")
    df_t3_mm_rep = _read_csv_if_exists(res_dir / "multimodal_relation_classification_reports.csv")
    df_t3_mm_cm = _read_csv_if_exists(res_dir / "multimodal_relation_confusion_matrices.csv")

    if df_t3_mm is not None and not df_t3_mm.empty:
        lines.extend([
            "---",
            "",
            "### 3.2 Multimodal Semiotic Relation Benchmark (`mm_test`, $N=129$ — `Stage A Raw` vs `Stage B Oracle` $\\times$ `raw` vs `std`)",
            "",
            "#### Master Comparison Table (All Multimodal Relation Runs)",
            _df_to_md(df_t3_mm[["Case", "Regime", "Prompting", "Model", "Scaling", "Macro F1", "Micro F1 (= Acc.)", "Weighted F1", "Additiva F1", "Divergente F1", "Parallela F1"]]),
            "",
        ])
        if df_t3_cls is not None and not df_t3_cls.empty:
            lines.extend([
                "#### Target Classes Per-Class F1 Breakdown (`ADD`, `PAR`, `DIV` on `mm_test`, $N=129$)",
                _df_to_md(df_t3_cls[["Case", "Regime", "Model", "Scaling", "ADD (n=68)", "PAR (n=23)", "DIV (n=38)", "Macro F1", "Micro F1 (= Acc.)"]]),
                "",
            ])

        lines.extend(["### 3.3 Multimodal Semiotic Relation Detailed Reports by Case (`Case 1` & `Case 2`)", ""])
        for case_id, case_group in df_t3_mm.groupby("Case", sort=False):
            reg_label = case_group.iloc[0]["Regime"]
            ckpt_name = case_group.iloc[0]["Checkpoint"]
            prompt_name = case_group.iloc[0]["Prompting"]
            lines.extend([
                f"#### {case_id}: {reg_label} (Checkpoint: `{ckpt_name}`, Eval Prompt: `{prompt_name}`)",
                "",
                _df_to_md(case_group[["Model", "Scaling", "Macro F1", "Micro F1 (= Acc.)", "Weighted F1", "Additiva F1", "Divergente F1", "Parallela F1"]]),
                "",
            ])
            for _, run_row in case_group.iterrows():
                m_name = run_row["Model"]
                sc = run_row["Scaling"]
                lines.append(f"##### Run: `{case_id}` | `{m_name}` | Scaling: `{sc}`")
                if df_t3_mm_rep is not None:
                    mask = (df_t3_mm_rep["Case"] == case_id) & (df_t3_mm_rep["Model"] == m_name) & (df_t3_mm_rep["Scaling"] == sc)
                    lines.extend(["**Classification Report**:", "", _df_to_md(df_t3_mm_rep[mask][["Label", "Precision", "Recall", "F1-Score", "Support"]]), ""])
                if df_t3_mm_cm is not None:
                    mask = (df_t3_mm_cm["Case"] == case_id) & (df_t3_mm_cm["Model"] == m_name) & (df_t3_mm_cm["Scaling"] == sc)
                    cm_cols = ["True_Label"] + [f"Pred_{c}" for c in RELATION_LABELS] + ["Total_True"]
                    lines.extend(["**3x3 Confusion Matrix**:", "", _df_to_md(df_t3_mm_cm[mask][cm_cols], floatfmt=".0f"), ""])

    out_file = res_dir / "ALL_EXPERIMENTS_COMPLETE_REPORT.md"
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved exhaustive complete report: {out_file}")


# ========================================================================================
# 2. Streamlined Paper Tables & LaTeX Builder (Unified 'raw' Protocol + Step 1 & Step 2 Raw)
# ========================================================================================
def generate_paper_artifacts(res_dir: Path):
    print("\n" + "=" * 80)
    print("GENERATING PAPER TABLES & COMPLETE EXPERIMENTAL REPORTS...")
    print("=" * 80)

    # First build the exhaustive ALL_EXPERIMENTS_COMPLETE_REPORT.md
    generate_all_experiments_complete_report(res_dir)

    # Load summaries for streamlined paper tables (unified 'raw' protocol)
    df_txt_emo = _read_csv_if_exists(res_dir / "train_textual_emotion_summary.csv")
    df_txt_iro = _read_csv_if_exists(res_dir / "train_textual_irony_summary.csv")
    df_mm_emo_base = _read_csv_if_exists(res_dir / "multimodal_emotion_baselines.csv")
    df_mm_emo_sub = _read_csv_if_exists(res_dir / "multimodal_emotion_subsets.csv")
    df_mm_iro_base = _read_csv_if_exists(res_dir / "multimodal_irony_baselines.csv")
    df_mm_iro_sub = _read_csv_if_exists(res_dir / "multimodal_irony_subsets.csv")
    df_mm_rel_base = _read_csv_if_exists(res_dir / "multimodal_relation_baselines.csv")
    df_mm_rel_cls = _read_csv_if_exists(res_dir / "multimodal_relation_target_classes.csv")

    baselines_rows = []

    # Table I: Emotion (Textual + Multimodal 'raw' across Step 1 Base, Step 2 Raw, Step 2 Oracle)
    if df_txt_emo is not None:
        for _, r in df_txt_emo.iterrows():
            baselines_rows.append({
                "Task": "Emotion Detection",
                "Section": "Textual Test Set (me_test, N=405)",
                "Configuration": f"Textual test set ({r['Regime']})",
                "Prompting": r["Prompt"],
                "Scaling": "—",
                "Macro F1": r["Macro F1"],
                "Micro F1 / Acc": r["Micro F1"],
            })
    if df_mm_emo_base is not None and not df_mm_emo_base.empty:
        sub_raw = df_mm_emo_base[df_mm_emo_base["Scaling"] == "raw"]
        for _, r in sub_raw.iterrows():
            baselines_rows.append({
                "Task": "Emotion Detection",
                "Section": "Multimodal Test Set (mm_test, N=129)",
                "Configuration": r["Configuration"],
                "Prompting": r["Prompting"],
                "Scaling": r["Scaling"],
                "Macro F1": r["Macro F1"],
                "Micro F1 / Acc": r["Micro F1"],
            })

    # Table IV: Irony (Textual + Multimodal 'raw' across Step 1 Base, Step 2 Raw, Step 2 Oracle)
    if df_txt_iro is not None:
        for _, r in df_txt_iro.iterrows():
            baselines_rows.append({
                "Task": "Irony Detection",
                "Section": "Textual Test Set (me_test, N=405)",
                "Configuration": f"Textual test set ({r['Regime']})",
                "Prompting": r["Prompt"],
                "Scaling": "—",
                "Macro F1": r["Macro F1"],
                "Micro F1 / Acc": r["Micro F1 (= Acc)"],
            })
    if df_mm_iro_base is not None and not df_mm_iro_base.empty:
        sub_raw = df_mm_iro_base[df_mm_iro_base["Scaling"] == "raw"]
        for _, r in sub_raw.iterrows():
            baselines_rows.append({
                "Task": "Irony Detection",
                "Section": "Multimodal Test Set (mm_test, N=129)",
                "Configuration": r["Configuration"],
                "Prompting": r["Prompting"],
                "Scaling": r["Scaling"],
                "Macro F1": r["Macro F1"],
                "Micro F1 / Acc": r["Micro F1 (= Acc.)"],
            })

    # Table VII: Semiotic Relation ('raw')
    if df_mm_rel_base is not None and not df_mm_rel_base.empty:
        sub_raw = df_mm_rel_base[df_mm_rel_base["Scaling"] == "raw"]
        for _, r in sub_raw.iterrows():
            baselines_rows.append({
                "Task": "Semiotic Relation",
                "Section": "Multimodal Test Set (mm_test, N=129)",
                "Configuration": r["Configuration"],
                "Prompting": r["Prompting"],
                "Scaling": r["Scaling"],
                "Macro F1": r["Macro F1"],
                "Micro F1 / Acc": r["Micro F1 (= Acc.)"],
            })

    df_baselines_all = pd.DataFrame(baselines_rows)
    df_baselines_all.to_csv(res_dir / "paper_baselines_summary.csv", index=False)

    # Unconfounded Ablation Matrix Table
    ablation_rows = []
    if df_txt_emo is not None:
        r0 = find_row(df_txt_emo, "Regime", "Step 1 Base")
        r1 = find_row(df_txt_emo, "Regime", "Step 2 Raw")
        r2 = find_row(df_txt_emo, "Regime", "Step 2 Oracle (Train: no rel)")
        r3 = find_row(df_txt_emo, "Regime", "Step 2 Oracle (Train: +rel)")
        if r0 is not None and r1 is not None and r2 is not None:
            row_ab = {
                "Task": "Emotion Detection (me_test, N=405)",
                "Step 1 Base (Raw)": f"{r0['Micro F1']:.4f} Micro / {r0['Macro F1']:.4f} Macro",
                "Step 2 Raw Ablation": f"{r1['Micro F1']:.4f} Micro / {r1['Macro F1']:.4f} Macro",
                "Step 2 Oracle (Train: no rel)": f"{r2['Micro F1']:.4f} Micro / {r2['Macro F1']:.4f} Macro",
                "Step 2 Oracle (Train: +rel)": f"{r3['Micro F1']:.4f} Micro / {r3['Macro F1']:.4f} Macro" if r3 is not None else "N/A",
                "Delta (Oracle no-rel vs Step 2 Raw)": f"{r2['Micro F1'] - r1['Micro F1']:+.4f} Micro / {r2['Macro F1'] - r1['Macro F1']:+.4f} Macro",
            }
            ablation_rows.append(row_ab)
    if df_txt_iro is not None:
        r0_i = find_row(df_txt_iro, "Regime", "Step 1 Base")
        r1_i = find_row(df_txt_iro, "Regime", "Step 2 Raw")
        r2_i = find_row(df_txt_iro, "Regime", "Step 2 Oracle (Train: no rel)")
        r3_i = find_row(df_txt_iro, "Regime", "Step 2 Oracle (Train: +rel)")
        if r0_i is not None and r1_i is not None and r2_i is not None:
            row_ab_i = {
                "Task": "Irony Detection (me_test, N=405)",
                "Step 1 Base (Raw)": f"{r0_i['Micro F1 (= Acc)']:.4f} Acc / {r0_i['Macro F1']:.4f} Macro",
                "Step 2 Raw Ablation": f"{r1_i['Micro F1 (= Acc)']:.4f} Acc / {r1_i['Macro F1']:.4f} Macro",
                "Step 2 Oracle (Train: no rel)": f"{r2_i['Micro F1 (= Acc)']:.4f} Acc / {r2_i['Macro F1']:.4f} Macro",
                "Step 2 Oracle (Train: +rel)": f"{r3_i['Micro F1 (= Acc)']:.4f} Acc / {r3_i['Macro F1']:.4f} Macro" if r3_i is not None else "N/A",
                "Delta (Oracle no-rel vs Step 2 Raw)": f"{r2_i['Micro F1 (= Acc)'] - r1_i['Micro F1 (= Acc)']:+.4f} Acc / {r2_i['Macro F1'] - r1_i['Macro F1']:+.4f} Macro",
            }
            ablation_rows.append(row_ab_i)
    df_ablation = pd.DataFrame(ablation_rows)
    df_ablation.to_csv(res_dir / "paper_ablation_matrix.csv", index=False)

    # Semiotic Subsets Combined Table (Dual-Encoder 'raw')
    sub_combined = []
    if df_mm_emo_sub is not None and not df_mm_emo_sub.empty:
        sub_e = df_mm_emo_sub[(df_mm_emo_sub["Scaling"] == "raw") & df_mm_emo_sub["Model"].astype(str).str.contains("Dual-Encoder", case=False, na=False)]
        for _, r in sub_e.iterrows():
            sub_combined.append({
                "Task": "Emotion Detection (Micro F1)",
                "Case": r["Case"],
                "Model Setting": r["Model Setting"],
                "Prompting": r["Prompting"],
                "Scaling": r["Scaling"],
                "ADD (N=68)": r["ADD (N=68)"],
                "PAR (N=23)": r["PAR (N=23)"],
                "DIV (N=38)": r["DIV (N=38)"],
                "Overall Macro F1": r["Macro F1"],
            })
    if df_mm_iro_sub is not None and not df_mm_iro_sub.empty:
        sub_i = df_mm_iro_sub[(df_mm_iro_sub["Scaling"] == "raw") & df_mm_iro_sub["Model"].astype(str).str.contains("Dual-Encoder", case=False, na=False)]
        for _, r in sub_i.iterrows():
            sub_combined.append({
                "Task": "Irony Detection (Accuracy / Micro F1)",
                "Case": r["Case"],
                "Model Setting": r["Model Setting"],
                "Prompting": r["Prompting"],
                "Scaling": r["Scaling"],
                "ADD (N=68)": r["ADD (N=68)"],
                "PAR (N=23)": r["PAR (N=23)"],
                "DIV (N=38)": r["DIV (N=38)"],
                "Overall Macro F1": r["Macro F1"],
            })
    df_subsets_all = pd.DataFrame(sub_combined)
    df_subsets_all.to_csv(res_dir / "paper_subsets_summary.csv", index=False)

    # Save Excel workbook
    try:
        with pd.ExcelWriter(res_dir / "paper_tables.xlsx") as writer:
            df_baselines_all.to_excel(writer, sheet_name="Baselines_Raw", index=False)
            df_subsets_all.to_excel(writer, sheet_name="Semiotic_Subsets_Raw", index=False)
            df_ablation.to_excel(writer, sheet_name="Ablation_Matrix", index=False)
            if df_mm_emo_base is not None:
                df_mm_emo_base.to_excel(writer, sheet_name="MM_Emotion_All", index=False)
            if df_mm_iro_base is not None:
                df_mm_iro_base.to_excel(writer, sheet_name="MM_Irony_All", index=False)
            if df_mm_rel_base is not None:
                df_mm_rel_base.to_excel(writer, sheet_name="MM_Relation_All", index=False)
        print(f"Saved Excel workbook: {res_dir / 'paper_tables.xlsx'}")
    except Exception as e:
        print(f"Excel export skipped: {e}")

    # Save Streamlined Markdown Tables (paper_summary_tables.md & PAPER_TABLES.md)
    df_rel_raw_cls = None
    if df_mm_rel_cls is not None and not df_mm_rel_cls.empty:
        df_rel_raw_cls = df_mm_rel_cls[df_mm_rel_cls["Scaling"] == "raw"]

    md_summary = f"""# MultiEMiT: Streamlined Publication Tables Summary (`raw` Unified Protocol)

> For the exhaustive case-by-case breakdown including `std` vs `raw`, `+rel` vs `no rel`, per-label `classification_report`, and confusion matrices, see `ALL_EXPERIMENTS_COMPLETE_REPORT.md`.

---

## 1. Baseline Discriminative Tables (`raw` Feature Protocol — Tables I, IV, VII)

{_df_to_md(df_baselines_all, floatfmt=".3f")}

---

## 2. Semiotic Subsets Micro F1 / Accuracy (`raw` Dual-Encoder — Tables III & VI)

{_df_to_md(df_subsets_all, floatfmt=".3f")}

---

## 3. The Unconfounded Textual Ablation Matrix (`me_test`, $N=405$)

{_df_to_md(df_ablation)}

---

## 4. Semiotic Relation Target Classes Breakdown (`raw` — Table IX)

{_df_to_md(df_rel_raw_cls, floatfmt=".3f")}
"""
    (res_dir / "paper_summary_tables.md").write_text(md_summary, encoding="utf-8")
    (res_dir / "PAPER_TABLES.md").write_text(md_summary, encoding="utf-8")
    print(f"Saved Streamlined Markdown tables: {res_dir / 'PAPER_TABLES.md'} & {res_dir / 'paper_summary_tables.md'}")

    print("\n" + "=" * 80)
    print("ALL PAPER & COMPLETE REPORT ARTIFACTS GENERATED SUCCESSFULLY.")
    print("=" * 80)


def main():
    setup_console()
    args = parse_args()
    res_dir = Path(args.results_dir).resolve()
    ckpt_dir = Path(args.checkpoints_dir).resolve()
    res_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not args.report_only:
        extra_textual = [
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--seed", str(args.seed),
            "--checkpoints-dir", str(ckpt_dir),
            "--results-dir", str(res_dir),
        ]
        if args.force_retrain:
            extra_textual.append("--force-retrain")

        extra_mm = [
            "--batch-size", str(args.eval_batch_size),
            "--seed", str(args.seed),
            "--checkpoints-dir", str(ckpt_dir),
            "--results-dir", str(res_dir),
        ]

        tasks_to_run = ["emotion", "irony", "relation"] if args.task == "all" else [args.task]

        for t in tasks_to_run:
            print(f"\n=======================================================")
            print(f"EXECUTING PIPELINE FOR: {t.upper()}")
            print(f"=======================================================")
            if t == "emotion":
                run_script("train_textual_emotion.py", extra_textual)
                run_script("evaluate_multimodal_emotion.py", extra_mm)
            elif t == "irony":
                run_script("train_textual_irony.py", extra_textual)
                run_script("evaluate_multimodal_irony.py", extra_mm)
            elif t == "relation":
                run_script("train_textual_relation.py", extra_textual)
                run_script("evaluate_multimodal_relation.py", extra_mm)

    generate_paper_artifacts(res_dir)


if __name__ == "__main__":
    main()
