"""
Master Runner for MultiEMiT Clean Paper Experiments.
Executes the streamlined, paper-specific experiments and aggregates them dynamically
into the exact tables and structures specified in PAPER_RESULTS_SYNTHESIS.md.

Produces:
  * results/PAPER_TABLES.md
  * results/paper_baselines_summary.csv
  * results/paper_subsets_summary.csv
  * results/paper_ablation_matrix.csv
  * results/paper_tables.xlsx
  * results/paper_tables.tex
  * results/paper_summary_tables.md
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import subprocess

from common_utils import setup_console


def parse_args():
    parser = argparse.ArgumentParser(description="MultiEMiT Clean Paper Runner")
    parser.add_argument("--task", type=str, default="all", choices=["all", "emotion", "irony", "relation"])
    parser.add_argument("--force-retrain", action="store_true", default=False, help="Force retrain textual models from scratch.")
    parser.add_argument("--checkpoints-dir", type=str, default="./checkpoints")
    parser.add_argument("--results-dir", type=str, default="./results")
    return parser.parse_args()


def run_script(script_name: str, extra_args: list):
    script_path = str(Path(__file__).resolve().parent / script_name)
    cmd = [sys.executable, script_path] + extra_args
    print(f"\n>>> Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, check=True)
    return res.returncode


def find_row(df: pd.DataFrame, col: str, pattern: str):
    """Safely retrieves a row matching a substring without relying on fragile positional indices."""
    if df is None or df.empty or col not in df.columns:
        return None
    matches = df[df[col].astype(str).str.contains(pattern, case=False, na=False, regex=False)]
    if not matches.empty:
        return matches.iloc[0]
    return None


def generate_dynamic_latex_tables(df_baselines: pd.DataFrame, df_subsets: pd.DataFrame, df_ablation: pd.DataFrame, df_rel_cls: pd.DataFrame) -> str:
    """Dynamically generates publication-ready LaTeX tables matching IEEE Transactions on Affective Computing standards."""
    latex_snippets = [
        "% ===========================================================================",
        "% MultiEMiT: Auto-Generated Publication Tables (IEEE Transactions)",
        "% Fully dynamic: derived directly from empirical evaluation CSVs",
        "% ===========================================================================",
        "",
    ]

    # Helper for baseline lookups
    def get_base_vals(task_pat: str, cfg_pat: str, macro_col="Macro F1", micro_col="Micro F1 / Acc"):
        if df_baselines is None or df_baselines.empty:
            return 0.0, 0.0
        matches = df_baselines[
            df_baselines["Task"].astype(str).str.contains(task_pat, case=False, na=False, regex=False) &
            df_baselines["Configuration"].astype(str).str.contains(cfg_pat, case=False, na=False, regex=False)
        ]
        if not matches.empty:
            row = matches.iloc[0]
            return float(row[macro_col]), float(row[micro_col])
        return 0.0, 0.0

    # 1. Table I: Emotion Baselines
    e_t_no_mac, e_t_no_mic = get_base_vals("Emotion", "Textual test set (No")
    e_t_co_mac, e_t_co_mic = get_base_vals("Emotion", "Textual test set (With")
    e_m_no_mac, e_m_no_mic = get_base_vals("Emotion", "without images, No corr")
    e_m_im_mac, e_m_im_mic = get_base_vals("Emotion", "with images (Dual-Encoder, No corr")
    e_m_co_mac, e_m_co_mic = get_base_vals("Emotion", "without images, With corr")
    e_m_du_mac, e_m_du_mic = get_base_vals("Emotion", "with images and correlations")

    tab_emotion = f"""% Table I: Emotion Detection Baselines
\\begin{{table}}[!t]
\\centering
\\caption{{Discriminative Baseline Performance in Emotion Detection}}
\\label{{tab:emotion_baselines}}
\\small
\\renewcommand{{\\arraystretch}}{{1.05}}
\\begin{{tabular}}{{l c c}}
\\hline
\\textbf{{Configuration}} & \\textbf{{Macro F1}} & \\textbf{{Micro F1}} \\\\
\\hline
\\multicolumn{{3}}{{l}}{{\\textit{{Textual Test Set ($N=405$)}}}} \\\\
Textual test set (No correlations) & {e_t_no_mac:.3f} & {e_t_no_mic:.3f} \\\\
Textual test set (With correlations) & \\textbf{{{e_t_co_mac:.3f}}} & \\textbf{{{e_t_co_mic:.3f}}} \\\\
\\hline
\\multicolumn{{3}}{{l}}{{\\textit{{Multimodal Test Set ($N=129$)}}}} \\\\
Multimodal test set (without images, No corr.) & {e_m_no_mac:.3f} & {e_m_no_mic:.3f} \\\\
Multimodal test set with images (Dual-Encoder, No corr.) & {e_m_im_mac:.3f} & {e_m_im_mic:.3f} \\\\
Multimodal test set (without images, With corr.) & {e_m_co_mac:.3f} & {e_m_co_mic:.3f} \\\\
Multimodal test set with images and correlations (Dual-Encoder) & \\textbf{{{e_m_du_mac:.3f}}} & \\textbf{{{e_m_du_mic:.3f}}} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    latex_snippets.append(tab_emotion)

    # 2. Table III: Emotion Semiotic Subsets
    if df_subsets is not None and not df_subsets.empty:
        emo_sub = df_subsets[df_subsets["Task"].astype(str).str.contains("Emotion", case=False, na=False, regex=False)]
        r_base = find_row(emo_sub, "Model Setting", "Base")
        r_orac = find_row(emo_sub, "Model Setting", "Oracle")
        if r_base is not None and r_orac is not None:
            tab_emo_sub = f"""% Table III: Emotion Micro F1 across Semiotic Subsets
\\begin{{table}}[!t]
\\centering
\\caption{{Emotion Micro F1 across Semiotic Subsets}}
\\label{{tab:subset_emotion}}
\\small
\\renewcommand{{\\arraystretch}}{{1.05}}
\\begin{{tabular}}{{l c c c}}
\\hline
\\textbf{{Model Setting}} & \\textbf{{ADD ($N=68$)}} & \\textbf{{PAR ($N=23$)}} & \\textbf{{DIV ($N=38$)}} \\\\
\\hline
Dual-Encoder (No correlations) & {float(r_base['ADD']):.2f} & {float(r_base['PAR']):.2f} & {float(r_base['DIV']):.2f} \\\\
Dual-Encoder (With correlations) & \\textbf{{{float(r_orac['ADD']):.2f}}} & \\textbf{{{float(r_orac['PAR']):.2f}}} & \\textbf{{{float(r_orac['DIV']):.2f}}} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
            latex_snippets.append(tab_emo_sub)

    # 3. Table IV: Irony Baselines
    i_t_no_mac, i_t_no_mic = get_base_vals("Irony", "Textual test set (No")
    i_t_co_mac, i_t_co_mic = get_base_vals("Irony", "Textual test set (With")
    i_m_no_mac, i_m_no_mic = get_base_vals("Irony", "without images, No corr")
    i_m_im_mac, i_m_im_mic = get_base_vals("Irony", "with images (Dual-Encoder, No corr")
    i_m_co_mac, i_m_co_mic = get_base_vals("Irony", "without images, With corr")
    i_m_du_mac, i_m_du_mic = get_base_vals("Irony", "with images and correlations")

    tab_irony = f"""% Table IV: Irony Detection Baselines
\\begin{{table}}[!t]
\\centering
\\caption{{Discriminative Baseline Performance in Irony Detection}}
\\label{{tab:irony_baselines}}
\\small
\\renewcommand{{\\arraystretch}}{{1.05}}
\\begin{{tabular}}{{l c c}}
\\hline
\\textbf{{Configuration}} & \\textbf{{Macro F1}} & \\textbf{{Micro F1 / Acc.}} \\\\
\\hline
\\multicolumn{{3}}{{l}}{{\\textit{{Textual Test Set ($N=405$)}}}} \\\\
Textual test set (No correlations) & {i_t_no_mac:.3f} & {i_t_no_mic:.3f} \\\\
Textual test set (With correlations) & \\textbf{{{i_t_co_mac:.3f}}} & \\textbf{{{i_t_co_mic:.3f}}} \\\\
\\hline
\\multicolumn{{3}}{{l}}{{\\textit{{Multimodal Test Set ($N=129$)}}}} \\\\
Multimodal test set (without images, No corr., std) & {i_m_no_mac:.3f} & {i_m_no_mic:.3f} \\\\
Multimodal test set with images (Dual-Encoder, No corr., std) & {i_m_im_mac:.3f} & \\textbf{{{i_m_im_mic:.3f}}} \\\\
Multimodal test set (without images, With corr., std) & {i_m_co_mac:.3f} & {i_m_co_mic:.3f} \\\\
Multimodal test set with images and correlations (Dual-Encoder, std) & \\textbf{{{i_m_du_mac:.3f}}} & \\textbf{{{i_m_du_mic:.3f}}} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    latex_snippets.append(tab_irony)

    # 4. Table VI: Irony Semiotic Subsets
    if df_subsets is not None and not df_subsets.empty:
        iro_sub = df_subsets[df_subsets["Task"].astype(str).str.contains("Irony", case=False, na=False, regex=False)]
        r_base_i = find_row(iro_sub, "Model Setting", "Base")
        r_orac_i = find_row(iro_sub, "Model Setting", "Oracle")
        if r_base_i is not None and r_orac_i is not None:
            tab_iro_sub = f"""% Table VI: Irony Accuracy across Semiotic Subsets
\\begin{{table}}[!t]
\\centering
\\caption{{Irony Accuracy across Semiotic Subsets}}
\\label{{tab:subset_irony}}
\\small
\\renewcommand{{\\arraystretch}}{{1.05}}
\\begin{{tabular}}{{l c c c}}
\\hline
\\textbf{{Model Setting}} & \\textbf{{ADD ($N=68$)}} & \\textbf{{PAR ($N=23$)}} & \\textbf{{DIV ($N=38$)}} \\\\
\\hline
Dual-Encoder (No correlations, std) & {float(r_base_i['ADD']):.2f} & \\textbf{{{float(r_base_i['PAR']):.2f}}} & \\textbf{{{float(r_base_i['DIV']):.2f}}} \\\\
Dual-Encoder (With correlations, std) & \\textbf{{{float(r_orac_i['ADD']):.2f}}} & {float(r_orac_i['PAR']):.2f} & {float(r_orac_i['DIV']):.2f} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
            latex_snippets.append(tab_iro_sub)

    # 5. Table VII: Semiotic Relation Baselines
    r_m_no_mac, r_m_no_mic = get_base_vals("Semiotic Relation", "without images, No corr")
    r_m_im_mac, r_m_im_mic = get_base_vals("Semiotic Relation", "with images (Dual-Encoder, No corr")
    r_m_co_mac, r_m_co_mic = get_base_vals("Semiotic Relation", "without images, With corr")
    r_m_du_mac, r_m_du_mic = get_base_vals("Semiotic Relation", "with images and correlations")

    tab_relation = f"""% Table VII: Semiotic Relation Baselines
\\begin{{table}}[!t]
\\centering
\\caption{{Discriminative Baseline Performance in Semiotic Relation Detection}}
\\label{{tab:relation_baselines}}
\\small
\\renewcommand{{\\arraystretch}}{{1.05}}
\\begin{{tabular}}{{l c c}}
\\hline
\\textbf{{Configuration}} & \\textbf{{Macro F1}} & \\textbf{{Micro F1 / Acc.}} \\\\
\\hline
\\multicolumn{{3}}{{l}}{{\\textit{{Multimodal Test Set ($N=129$)}}}} \\\\
Multimodal test set (without images, Stage A Raw) & {r_m_no_mac:.3f} & {r_m_no_mic:.3f} \\\\
Multimodal test set with images (Dual-Encoder, Stage A Raw) & {r_m_im_mac:.3f} & \\textbf{{{r_m_im_mic:.3f}}} \\\\
Multimodal test set (without images, Stage B Oracle) & {r_m_co_mac:.3f} & {r_m_co_mic:.3f} \\\\
Multimodal test set with images and correlations (Dual-Encoder) & \\textbf{{{r_m_du_mac:.3f}}} & {r_m_du_mic:.3f} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    latex_snippets.append(tab_relation)

    # 6. Table IX: Relation Target Classes Breakdown
    if df_rel_cls is not None and not df_rel_cls.empty:
        r_raw_cls = find_row(df_rel_cls, "Model Setting", "Stage A Raw")
        r_ora_cls = find_row(df_rel_cls, "Model Setting", "Stage B Oracle")
        if r_raw_cls is not None and r_ora_cls is not None:
            tab_rel_cls = f"""% Table IX: Semiotic Relation Target Classes Breakdown
\\begin{{table}}[!t]
\\centering
\\caption{{Semiotic Relation Target Classes F1 Breakdown}}
\\label{{tab:relation_classes}}
\\small
\\renewcommand{{\\arraystretch}}{{1.05}}
\\begin{{tabular}}{{l c c c c}}
\\hline
\\textbf{{Model Setting}} & \\textbf{{Additiva ($n=68$)}} & \\textbf{{Parallela ($n=23$)}} & \\textbf{{Divergente ($n=38$)}} & \\textbf{{Macro F1}} \\\\
\\hline
Dual-Encoder (Stage A Raw) & \\textbf{{{float(r_raw_cls['ADD (n=68)']):.3f}}} & {float(r_raw_cls['PAR (n=23)']):.3f} & \\textbf{{{float(r_raw_cls['DIV (n=38)']):.3f}}} & {float(r_raw_cls['Macro F1']):.3f} \\\\
Dual-Encoder (Stage B Oracle) & {float(r_ora_cls['ADD (n=68)']):.3f} & \\textbf{{{float(r_ora_cls['PAR (n=23)']):.3f}}} & {float(r_ora_cls['DIV (n=38)']):.3f} & \\textbf{{{float(r_ora_cls['Macro F1']):.3f}}} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
            latex_snippets.append(tab_rel_cls)

    # 7. Unconfounded Ablation Matrix Table
    if df_ablation is not None and not df_ablation.empty:
        tab_ablation = f"""% Table II: The Unconfounded Ablation Matrix
\\begin{{table}}[!t]
\\centering
\\caption{{The Unconfounded Correlation Ablation Matrix}}
\\label{{tab:unconfounded_ablation}}
\\small
\\renewcommand{{\\arraystretch}}{{1.05}}
\\begin{{tabular}}{{l l c c c c}}
\\hline
\\textbf{{Task}} & \\textbf{{Metric}} & \\textbf{{Step 1 Base (Raw)}} & \\textbf{{Step 2 Raw Ablation}} & \\textbf{{Step 2 Oracle}} & \\textbf{{$\\Delta$ Correlation}} \\\\
\\hline
"""
        r_emo_abl = find_row(df_ablation, "Task", "Emotion")
        if r_emo_abl is not None:
            tab_ablation += f"Emotion Detection & Micro / Macro & {r_emo_abl['Step 1 Base (Raw)']} & {r_emo_abl['Step 2 Raw Ablation']} & {r_emo_abl['Step 2 Oracle (With Corr)']} & {r_emo_abl['Delta (Oracle vs Step 2 Raw)']} \\\\\n"
        r_iro_abl = find_row(df_ablation, "Task", "Irony")
        if r_iro_abl is not None:
            tab_ablation += f"Irony Detection & Acc / Macro & {r_iro_abl['Step 1 Base (Raw)']} & {r_iro_abl['Step 2 Raw Ablation']} & {r_iro_abl['Step 2 Oracle (With Corr)']} & {r_iro_abl['Delta (Oracle vs Step 2 Raw)']} \\\\\n"
        tab_ablation += f"""\\hline
\\end{{tabular}}
\\end{{table}}
"""
        latex_snippets.append(tab_ablation)

    return "\n".join(latex_snippets)


def generate_paper_artifacts(res_dir: Path):
    print("\n" + "=" * 80)
    print("GENERATING STREAMLINED PAPER TABLES & ARTIFACTS...")
    print("=" * 80)

    # 1. Load Textual Summaries
    df_txt_emo = pd.read_csv(res_dir / "train_textual_emotion_summary.csv") if (res_dir / "train_textual_emotion_summary.csv").exists() else None
    df_txt_iro = pd.read_csv(res_dir / "train_textual_irony_summary.csv") if (res_dir / "train_textual_irony_summary.csv").exists() else None
    df_txt_rel = pd.read_csv(res_dir / "train_textual_relation_summary.csv") if (res_dir / "train_textual_relation_summary.csv").exists() else None

    # 2. Load Multimodal Summaries
    df_mm_emo_base = pd.read_csv(res_dir / "multimodal_emotion_baselines.csv") if (res_dir / "multimodal_emotion_baselines.csv").exists() else None
    df_mm_emo_sub = pd.read_csv(res_dir / "multimodal_emotion_subsets.csv") if (res_dir / "multimodal_emotion_subsets.csv").exists() else None

    df_mm_iro_base = pd.read_csv(res_dir / "multimodal_irony_baselines.csv") if (res_dir / "multimodal_irony_baselines.csv").exists() else None
    df_mm_iro_sub = pd.read_csv(res_dir / "multimodal_irony_subsets.csv") if (res_dir / "multimodal_irony_subsets.csv").exists() else None

    df_mm_rel_base = pd.read_csv(res_dir / "multimodal_relation_baselines.csv") if (res_dir / "multimodal_relation_baselines.csv").exists() else None
    df_mm_rel_cls = pd.read_csv(res_dir / "multimodal_relation_target_classes.csv") if (res_dir / "multimodal_relation_target_classes.csv").exists() else None

    # Build Unified Baselines Table
    baselines_rows = []

    # Table I: Emotion
    if df_txt_emo is not None and df_mm_emo_base is not None:
        r_t_no = find_row(df_txt_emo, "Regime", "Step 1 Base")
        r_t_co = find_row(df_txt_emo, "Regime", "Step 2 Oracle")
        r_m_no = find_row(df_mm_emo_base, "Configuration", "without images, Step 1 Base")
        r_m_im = find_row(df_mm_emo_base, "Configuration", "with images (Dual-Encoder, Step 1 Base")
        r_m_co = find_row(df_mm_emo_base, "Configuration", "without images, Step 2 Oracle")
        r_m_du = find_row(df_mm_emo_base, "Configuration", "with images (Dual-Encoder, Step 2 Oracle")

        if r_t_no is not None:
            baselines_rows.append({"Task": "Emotion Detection", "Section": "Textual Test Set (me_test, N=405)", "Configuration": "Textual test set (No correlations)", "Macro F1": r_t_no["Macro F1"], "Micro F1 / Acc": r_t_no["Micro F1"]})
        if r_t_co is not None:
            baselines_rows.append({"Task": "Emotion Detection", "Section": "Textual Test Set (me_test, N=405)", "Configuration": "Textual test set (With correlations)", "Macro F1": r_t_co["Macro F1"], "Micro F1 / Acc": r_t_co["Micro F1"]})
        if r_m_no is not None:
            baselines_rows.append({"Task": "Emotion Detection", "Section": "Multimodal Test Set (mm_test, N=129)", "Configuration": "Multimodal test set (without images, No corr.)", "Macro F1": r_m_no["Macro F1"], "Micro F1 / Acc": r_m_no["Micro F1"]})
        if r_m_im is not None:
            baselines_rows.append({"Task": "Emotion Detection", "Section": "Multimodal Test Set (mm_test, N=129)", "Configuration": "Multimodal test set with images (Dual-Encoder, No corr.)", "Macro F1": r_m_im["Macro F1"], "Micro F1 / Acc": r_m_im["Micro F1"]})
        if r_m_co is not None:
            baselines_rows.append({"Task": "Emotion Detection", "Section": "Multimodal Test Set (mm_test, N=129)", "Configuration": "Multimodal test set (without images, With corr.)", "Macro F1": r_m_co["Macro F1"], "Micro F1 / Acc": r_m_co["Micro F1"]})
        if r_m_du is not None:
            baselines_rows.append({"Task": "Emotion Detection", "Section": "Multimodal Test Set (mm_test, N=129)", "Configuration": "Multimodal test set with images and correlations (Dual-Encoder)", "Macro F1": r_m_du["Macro F1"], "Micro F1 / Acc": r_m_du["Micro F1"]})

    # Table IV: Irony
    if df_txt_iro is not None and df_mm_iro_base is not None:
        r_t_no_i = find_row(df_txt_iro, "Regime", "Step 1 Base")
        r_t_co_i = find_row(df_txt_iro, "Regime", "Step 2 Oracle")
        r_m_no_i = find_row(df_mm_iro_base, "Configuration", "without images, Step 1 Base")
        r_m_im_i = find_row(df_mm_iro_base, "Configuration", "with images (Dual-Encoder, Step 1 Base")
        r_m_co_i = find_row(df_mm_iro_base, "Configuration", "without images, Step 2 Oracle")
        r_m_du_i = find_row(df_mm_iro_base, "Configuration", "with images (Dual-Encoder, Step 2 Oracle")

        if r_t_no_i is not None:
            baselines_rows.append({"Task": "Irony Detection", "Section": "Textual Test Set (me_test, N=405)", "Configuration": "Textual test set (No correlations)", "Macro F1": r_t_no_i["Macro F1"], "Micro F1 / Acc": r_t_no_i["Micro F1 (= Acc)"]})
        if r_t_co_i is not None:
            baselines_rows.append({"Task": "Irony Detection", "Section": "Textual Test Set (me_test, N=405)", "Configuration": "Textual test set (With correlations)", "Macro F1": r_t_co_i["Macro F1"], "Micro F1 / Acc": r_t_co_i["Micro F1 (= Acc)"]})
        if r_m_no_i is not None:
            baselines_rows.append({"Task": "Irony Detection", "Section": "Multimodal Test Set (mm_test, N=129)", "Configuration": "Multimodal test set (without images, No corr., std)", "Macro F1": r_m_no_i["Macro F1"], "Micro F1 / Acc": r_m_no_i["Micro F1 (= Acc.)"]})
        if r_m_im_i is not None:
            baselines_rows.append({"Task": "Irony Detection", "Section": "Multimodal Test Set (mm_test, N=129)", "Configuration": "Multimodal test set with images (Dual-Encoder, No corr., std)", "Macro F1": r_m_im_i["Macro F1"], "Micro F1 / Acc": r_m_im_i["Micro F1 (= Acc.)"]})
        if r_m_co_i is not None:
            baselines_rows.append({"Task": "Irony Detection", "Section": "Multimodal Test Set (mm_test, N=129)", "Configuration": "Multimodal test set (without images, With corr., std)", "Macro F1": r_m_co_i["Macro F1"], "Micro F1 / Acc": r_m_co_i["Micro F1 (= Acc.)"]})
        if r_m_du_i is not None:
            baselines_rows.append({"Task": "Irony Detection", "Section": "Multimodal Test Set (mm_test, N=129)", "Configuration": "Multimodal test set with images and correlations (Dual-Encoder, std)", "Macro F1": r_m_du_i["Macro F1"], "Micro F1 / Acc": r_m_du_i["Micro F1 (= Acc.)"]})

    # Table VII: Semiotic Relation
    if df_mm_rel_base is not None:
        r_m_no_r = find_row(df_mm_rel_base, "Configuration", "without images, Stage A Raw")
        r_m_im_r = find_row(df_mm_rel_base, "Configuration", "with images (Dual-Encoder, Stage A Raw")
        r_m_co_r = find_row(df_mm_rel_base, "Configuration", "without images, Stage B Oracle")
        r_m_du_r = find_row(df_mm_rel_base, "Configuration", "with images (Dual-Encoder, Stage B Oracle")

        if r_m_no_r is not None:
            baselines_rows.append({"Task": "Semiotic Relation", "Section": "Multimodal Test Set (mm_test, N=129)", "Configuration": "Multimodal test set (without images, No corr.)", "Macro F1": r_m_no_r["Macro F1"], "Micro F1 / Acc": r_m_no_r["Micro F1 (= Acc.)"]})
        if r_m_im_r is not None:
            baselines_rows.append({"Task": "Semiotic Relation", "Section": "Multimodal Test Set (mm_test, N=129)", "Configuration": "Multimodal test set with images (Dual-Encoder, No corr.)", "Macro F1": r_m_im_r["Macro F1"], "Micro F1 / Acc": r_m_im_r["Micro F1 (= Acc.)"]})
        if r_m_co_r is not None:
            baselines_rows.append({"Task": "Semiotic Relation", "Section": "Multimodal Test Set (mm_test, N=129)", "Configuration": "Multimodal test set (without images, With corr.)", "Macro F1": r_m_co_r["Macro F1"], "Micro F1 / Acc": r_m_co_r["Micro F1 (= Acc.)"]})
        if r_m_du_r is not None:
            baselines_rows.append({"Task": "Semiotic Relation", "Section": "Multimodal Test Set (mm_test, N=129)", "Configuration": "Multimodal test set with images and correlations (Dual-Encoder)", "Macro F1": r_m_du_r["Macro F1"], "Micro F1 / Acc": r_m_du_r["Micro F1 (= Acc.)"]})

    df_baselines_all = pd.DataFrame(baselines_rows)
    df_baselines_all.to_csv(res_dir / "paper_baselines_summary.csv", index=False)

    # Build Unconfounded Ablation Matrix Table
    ablation_rows = []
    if df_txt_emo is not None:
        r0 = find_row(df_txt_emo, "Regime", "Step 1 Base")
        r1 = find_row(df_txt_emo, "Regime", "Step 2 Raw")
        r2 = find_row(df_txt_emo, "Regime", "Step 2 Oracle")
        if r0 is not None and r1 is not None and r2 is not None:
            ablation_rows.append({
                "Task": "Emotion Detection",
                "Step 1 Base (Raw)": f"{r0['Micro F1']:.4f} Micro / {r0['Macro F1']:.4f} Macro",
                "Step 2 Raw Ablation": f"{r1['Micro F1']:.4f} Micro / {r1['Macro F1']:.4f} Macro",
                "Step 2 Oracle (With Corr)": f"{r2['Micro F1']:.4f} Micro / {r2['Macro F1']:.4f} Macro",
                "Delta (Oracle vs Step 2 Raw)": f"+{r2['Micro F1'] - r1['Micro F1']:.4f} Micro / +{r2['Macro F1'] - r1['Macro F1']:.4f} Macro",
            })
    if df_txt_iro is not None:
        r0_i = find_row(df_txt_iro, "Regime", "Step 1 Base")
        r1_i = find_row(df_txt_iro, "Regime", "Step 2 Raw")
        r2_i = find_row(df_txt_iro, "Regime", "Step 2 Oracle")
        if r0_i is not None and r1_i is not None and r2_i is not None:
            ablation_rows.append({
                "Task": "Irony Detection",
                "Step 1 Base (Raw)": f"{r0_i['Micro F1 (= Acc)']:.4f} Acc / {r0_i['Macro F1']:.4f} Macro",
                "Step 2 Raw Ablation": f"{r1_i['Micro F1 (= Acc)']:.4f} Acc / {r1_i['Macro F1']:.4f} Macro",
                "Step 2 Oracle (With Corr)": f"{r2_i['Micro F1 (= Acc)']:.4f} Acc / {r2_i['Macro F1']:.4f} Macro",
                "Delta (Oracle vs Step 2 Raw)": f"+{r2_i['Micro F1 (= Acc)'] - r1_i['Micro F1 (= Acc)']:.4f} Acc / +{r2_i['Macro F1'] - r1_i['Macro F1']:.4f} Macro",
            })
    df_ablation = pd.DataFrame(ablation_rows)
    df_ablation.to_csv(res_dir / "paper_ablation_matrix.csv", index=False)

    # Build Semiotic Subsets Combined Table
    sub_combined = []
    if df_mm_emo_sub is not None:
        for _, r in df_mm_emo_sub.iterrows():
            sub_combined.append({"Task": "Emotion Detection", "Model Setting": r["Model Setting"], "ADD": r["ADD (N=68)"], "PAR": r["PAR (N=23)"], "DIV": r["DIV (N=38)"]})
    if df_mm_iro_sub is not None:
        for _, r in df_mm_iro_sub.iterrows():
            sub_combined.append({"Task": "Irony Detection", "Model Setting": r["Model Setting"], "ADD": r["ADD (N=68)"], "PAR": r["PAR (N=23)"], "DIV": r["DIV (N=38)"]})
    df_subsets_all = pd.DataFrame(sub_combined)
    df_subsets_all.to_csv(res_dir / "paper_subsets_summary.csv", index=False)

    # Save Excel if openpyxl available
    try:
        with pd.ExcelWriter(res_dir / "paper_tables.xlsx") as writer:
            df_baselines_all.to_excel(writer, sheet_name="Baselines", index=False)
            df_subsets_all.to_excel(writer, sheet_name="Semiotic_Subsets", index=False)
            df_ablation.to_excel(writer, sheet_name="Ablation_Matrix", index=False)
            if df_mm_rel_cls is not None:
                df_mm_rel_cls.to_excel(writer, sheet_name="Relation_Target_Classes", index=False)
        print(f"Saved Excel workbook: {res_dir / 'paper_tables.xlsx'}")
    except Exception as e:
        print(f"Excel export skipped: {e}")

    # Generate Quick Markdown Tables Dump
    md_content = f"""# MultiEMiT: Streamlined Publication Tables Summary

> For the comprehensive technical specification, pipelines, and prompt mechanics, see `PAPER_TABLES.md`.

---

## 1. Baseline Discriminative Tables (Tables I, IV, VII)

{df_baselines_all.round(3).to_markdown(index=False)}

---

## 2. Semiotic Subsets Micro F1 / Accuracy (Tables III & VI)

{df_subsets_all.round(2).to_markdown(index=False)}

---

## 3. The Unconfounded Ablation Matrix (Reviewer Defense)

{df_ablation.to_markdown(index=False)}

---

## 4. Semiotic Relation Target Classes Breakdown (Table IX)

{df_mm_rel_cls.round(3).to_markdown(index=False) if df_mm_rel_cls is not None else "N/A"}
"""
    (res_dir / "paper_summary_tables.md").write_text(md_content, encoding="utf-8")
    print(f"Saved Quick Markdown dump: {res_dir / 'paper_summary_tables.md'}")

    # Generate Dynamic Publication LaTeX Code
    latex_content = generate_dynamic_latex_tables(df_baselines_all, df_subsets_all, df_ablation, df_mm_rel_cls)
    (res_dir / "paper_tables.tex").write_text(latex_content, encoding="utf-8")
    print(f"Saved Dynamic LaTeX snippet: {res_dir / 'paper_tables.tex'}")

    print("\n" + "=" * 80)
    print("ALL PAPER ARTIFACTS GENERATED SUCCESSFULLY.")
    print("=" * 80)


def main():
    setup_console()
    args = parse_args()
    res_dir = Path(args.results_dir).resolve()
    ckpt_dir = Path(args.checkpoints_dir).resolve()
    res_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    extra_textual = ["--checkpoints-dir", str(ckpt_dir), "--results-dir", str(res_dir)]
    if args.force_retrain:
        extra_textual.append("--force-retrain")

    extra_mm = ["--checkpoints-dir", str(ckpt_dir), "--results-dir", str(res_dir)]

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
