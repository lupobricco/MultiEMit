"""
One-Click Experiment Runner & Result Aggregator for MultiEMit (Textual & Multimodal Dual-Encoder).

Executes all textual and multimodal notebooks in dependency order (each in an isolated Python
subprocess so GPU VRAM is 100% freed between notebooks) and compiles all metrics and per-label
classification reports into a single unified output in `./results/`:
  - `./results/all_experiments_summary.csv`
  - `./results/all_experiments_summary.xlsx`
  - `./results/all_experiments_full_report.md`
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

BERT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BERT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Ordered pipeline: Textual encoders train first (saving ./checkpoints/*),
# followed by Multimodal Dual-Encoder (FEEL-IT / UmBERTo + ViT-224) notebooks.
TEXTUAL_NOTEBOOKS = [
    ("train_textual_emotion.ipynb", "Task 1: Textual Emotion Detection (FEEL-IT)"),
    ("train_textual_irony.ipynb", "Task 2: Textual Irony & Sarcasm Detection (UmBERTo)"),
    ("train_textual_relation.ipynb", "Task 3: Textual Semiotic Relation (UmBERTo)"),
]

MULTIMODAL_NOTEBOOKS = [
    ("multimodal_dual_encoder_emotion.ipynb", "Task 1: Multimodal Emotion Dual-Encoder (FEEL-IT + ViT-224)"),
    ("multimodal_dual_encoder_irony.ipynb", "Task 2: Multimodal Irony Dual-Encoder (UmBERTo + ViT-224)"),
    ("multimodal_dual_encoder_relation.ipynb", "Task 3: Multimodal Semiotic Relation Dual-Encoder (UmBERTo + ViT-224)"),
]

SUMMARY_ORDER = [
    "train_textual_emotion",
    "multimodal_dual_encoder_emotion",
    "train_textual_irony",
    "multimodal_dual_encoder_irony",
    "train_textual_relation",
    "multimodal_dual_encoder_relation",
]


def build_runner_script(nb_path: Path, force_retrain_override=None) -> str:
    """
    Generates a self-contained Python script from a notebook's code cells,
    wrapped in `if __name__ == '__main__':` for Windows multiprocessing safety,
    replacing IPython `display(...)` with `print(...)` and stripping `%pip` magic lines.
    """
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    body_lines = [
        "import os, sys",
        "os.environ['TQDM_NOTEBOOK'] = 'false'",
        "def display(obj):",
        "    print(obj.to_string() if hasattr(obj, 'to_string') else obj)",
    ]

    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        raw_lines = "".join(cell.get("source", [])).splitlines()
        cleaned_lines = []
        for line in raw_lines:
            if line.strip().startswith("%pip") or line.strip().startswith("!pip"):
                continue
            if "dataloader_num_workers" in line or "dataloader_prefetch_factor" in line:
                continue
            if force_retrain_override is not None and line.strip().startswith("FORCE_RETRAIN"):
                val_str = "True" if force_retrain_override else "False"
                line = f"FORCE_RETRAIN = {val_str}"
            cleaned_lines.append(line)
        cell_code = "\n".join(cleaned_lines).strip()
        if cell_code:
            body_lines.append(f"# --- Cell {idx} ---")
            body_lines.extend(cleaned_lines)
            body_lines.append("")

    indented_body = "\n".join("    " + ln if ln else "" for ln in body_lines)
    return "if __name__ == '__main__':\n" + indented_body + "\n"


def run_notebook(nb_filename: str, label: str, force_retrain_override=None) -> dict:
    nb_path = BERT_DIR / nb_filename
    stem = nb_path.stem
    print("\n" + "=" * 88)
    print(f"[RUNNING] {label} ({nb_filename})")
    print("=" * 88, flush=True)

    script_code = build_runner_script(nb_path, force_retrain_override=force_retrain_override)
    temp_script = RESULTS_DIR / f"_runner_{stem}.py"
    temp_script.write_text(script_code, encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["TQDM_NOTEBOOK"] = "false"

    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, "-u", str(temp_script)],
        cwd=str(BERT_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    log_lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        log_lines.append(line)

    proc.wait()
    elapsed = time.time() - t0

    try:
        temp_script.unlink()
    except OSError:
        pass

    log_path = RESULTS_DIR / f"{stem}_execution.log"
    log_path.write_text("".join(log_lines), encoding="utf-8")

    status = "OK" if proc.returncode == 0 else f"FAILED (exit code {proc.returncode})"
    print(f"\n--> Finished {nb_filename} [{status}] in {elapsed:.1f}s")
    return {
        "notebook": nb_filename,
        "label": label,
        "status": status,
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 1),
    }


def df_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No results available._"
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = [header, sep]
    for _, row in df.iterrows():
        formatted = []
        for c in cols:
            val = row[c]
            if isinstance(val, float):
                formatted.append(f"{val:.4f}")
            else:
                formatted.append(str(val))
        rows.append("| " + " | ".join(formatted) + " |")
    return "\n".join(rows)


def aggregate_all_results() -> pd.DataFrame:
    """
    Reads all `./results/*_summary.csv` and `./results/*_reports.txt` files and writes:
      1. `./results/all_experiments_summary.csv`
      2. `./results/all_experiments_summary.xlsx`
      3. `./results/all_experiments_full_report.md`
    """
    summary_dfs = []
    reports_sections = []

    for stem in SUMMARY_ORDER:
        csv_path = RESULTS_DIR / f"{stem}_summary.csv"
        rep_path = RESULTS_DIR / f"{stem}_reports.txt"
        if csv_path.exists():
            df_part = pd.read_csv(csv_path)
            summary_dfs.append(df_part)
        if rep_path.exists():
            rep_text = rep_path.read_text(encoding="utf-8").strip()
            reports_sections.append((f"{stem}.ipynb", rep_text))

    if not summary_dfs:
        print("No summary CSV files found in ./results/ yet.")
        return pd.DataFrame()

    master_df = pd.concat(summary_dfs, ignore_index=True)
    metric_cols = ["Micro F1", "Macro F1", "Weighted F1", "Accuracy / Subset Acc"]
    for c in metric_cols:
        if c in master_df.columns:
            master_df[c] = master_df[c].round(4)

    # 1. Save single unified CSV
    master_csv_path = RESULTS_DIR / "all_experiments_summary.csv"
    master_df.to_csv(master_csv_path, index=False)

    # 2. Save multi-sheet Excel file
    master_xlsx_path = RESULTS_DIR / "all_experiments_summary.xlsx"
    try:
        with pd.ExcelWriter(master_xlsx_path) as writer:
            master_df.to_excel(writer, sheet_name="All_Experiments", index=False)
            for sheet_name, keyword in [
                ("Task1_Emotion", "Emotion"),
                ("Task2_Irony", "Irony"),
                ("Task3_Relation", "Relation"),
            ]:
                sub = master_df[master_df["Task"].str.contains(keyword, case=False, na=False)]
                if not sub.empty:
                    sub.to_excel(writer, sheet_name=sheet_name, index=False)
    except Exception as exc:
        print(f"Note: Could not write Excel file ({exc}); CSV and Markdown reports were saved.")

    # 3. Save single comprehensive Markdown report
    md_lines = [
        "# MultiEMit — Unified Experimental Results Report",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        "**Datasets:** `datasets/corpora/correctly_split/` (`textual_only_cleaned.csv`, `multimodal_train_seed42.csv`, `multimodal_test_seed42.csv`)",
        "",
        "## 1. Master Comparison Table (All Tasks & Modalities)",
        "",
        df_to_markdown_table(master_df),
        "",
    ]

    for title, keyword in [
        ("## 2. Task 1: Emotion Detection (10-Label Multi-Label)", "Emotion"),
        ("## 3. Task 2: Irony & Sarcasm Detection (3-Class Single-Label)", "Irony"),
        ("## 4. Task 3: Semiotic Relation Classification (3-Class Single-Label)", "Relation"),
    ]:
        sub = master_df[master_df["Task"].str.contains(keyword, case=False, na=False)].drop(columns=["Task"])
        if not sub.empty:
            md_lines.extend([title, "", df_to_markdown_table(sub), ""])

    if reports_sections:
        md_lines.extend([
            "---",
            "",
            "## 5. Per-Class Classification Reports (All Notebooks)",
            "",
        ])
        for nb_name, rep_text in reports_sections:
            md_lines.extend([
                f"### `{nb_name}`",
                "```text",
                rep_text,
                "```",
                "",
            ])

    master_md_path = RESULTS_DIR / "all_experiments_full_report.md"
    master_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print("\n" + "=" * 88)
    print("UNIFIED RESULTS SAVED TO:")
    print(f"  1. CSV Table       : {master_csv_path}")
    print(f"  2. Excel Workbook  : {master_xlsx_path}")
    print(f"  3. Full MD Report  : {master_md_path}")
    print("=" * 88)
    return master_df


def main():
    parser = argparse.ArgumentParser(description="Run all MultiEMit experiments and aggregate results.")
    parser.add_argument(
        "--only-aggregate",
        action="store_true",
        help="Only aggregate existing ./results/*_summary.csv files without running notebooks.",
    )
    parser.add_argument(
        "--skip-textual",
        action="store_true",
        help="Skip textual encoder training notebooks and only run Multimodal Dual-Encoder notebooks.",
    )
    parser.add_argument(
        "--skip-multimodal",
        action="store_true",
        help="Skip Multimodal Dual-Encoder notebooks and only run textual encoder notebooks.",
    )
    parser.add_argument(
        "--force-retrain",
        choices=["true", "false"],
        default=None,
        help="Override FORCE_RETRAIN in all executed notebooks ('true' or 'false').",
    )
    parser.add_argument(
        "--notebooks",
        nargs="*",
        default=None,
        help="Optional explicit list of notebook filenames to run.",
    )
    args = parser.parse_args()

    force_override = None
    if args.force_retrain is not None:
        force_override = args.force_retrain.lower() == "true"

    if not args.only_aggregate:
        pipeline_to_run = []
        if args.notebooks:
            for nb_name in args.notebooks:
                pipeline_to_run.append((nb_name, nb_name))
        else:
            if not args.skip_textual:
                pipeline_to_run.extend(TEXTUAL_NOTEBOOKS)
            if not args.skip_multimodal:
                pipeline_to_run.extend(MULTIMODAL_NOTEBOOKS)

        run_summary = []
        for nb_file, label in pipeline_to_run:
            info = run_notebook(nb_file, label, force_retrain_override=force_override)
            run_summary.append(info)

        print("\n=== Execution Status Summary ===")
        print(pd.DataFrame(run_summary).to_string(index=False))

    master_df = aggregate_all_results()
    if not master_df.empty:
        print("\n=== Master Comparison Table (All Tasks & Configurations) ===")
        print(master_df.to_string(index=False))


if __name__ == "__main__":
    main()

