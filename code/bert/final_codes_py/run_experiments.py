"""
Master Experiment Runner & Result Aggregator for MultiEMit.

Provides easy parameter access and CLI flags to control exactly which tasks
(Emotion, Irony, Semiotic Relation) and stages (Textual, Multimodal) to execute.

Each component runs in an isolated Python subprocess so PyTorch CUDA VRAM is
100% freed between stages without cache fragmentation.

Examples:
  # Run everything (all 3 tasks, textual + multimodal):
  python run_experiments.py

  # Train only textual models for emotion and irony:
  python run_experiments.py --tasks emotion,irony --stages textual

  # Run only multimodal evaluation:
  python run_experiments.py --stages multimodal

  # Force retraining with custom hyperparameters:
  python run_experiments.py --tasks irony --force-retrain --epochs 10 --batch-size 16 --lr 2e-5

  # Dry-run to preview execution order:
  python run_experiments.py --dry-run
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
CHECKPOINTS_DIR = SCRIPT_DIR / "checkpoints"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

TASK_SCRIPTS = {
    "emotion": {
        "textual": ("train_textual_emotion.py", "Task 1: Textual Emotion Detection (FEEL-IT)"),
        "multimodal": ("multimodal_dual_encoder_emotion.py", "Task 1: Multimodal Emotion Dual-Encoder (FEEL-IT + ViT-224)"),
    },
    "irony": {
        "textual": ("train_textual_irony.py", "Task 2: Textual Irony Detection (UmBERTo)"),
        "multimodal": ("multimodal_dual_encoder_irony.py", "Task 2: Multimodal Irony Dual-Encoder (UmBERTo + ViT-224)"),
    },
    "relation": {
        "textual": ("train_textual_relation.py", "Task 3: Textual Semiotic Relation (UmBERTo)"),
        "multimodal": ("multimodal_dual_encoder_relation.py", "Task 3: Multimodal Semiotic Relation Dual-Encoder (UmBERTo + ViT-224)"),
    },
}

ALL_TASKS = ["emotion", "irony", "relation"]
ALL_STAGES = ["textual", "multimodal"]

SUMMARY_FILE_ORDER = [
    "train_textual_emotion_summary.csv",
    "multimodal_dual_encoder_emotion_summary.csv",
    "train_textual_irony_summary.csv",
    "multimodal_dual_encoder_irony_summary.csv",
    "train_textual_relation_summary.csv",
    "multimodal_dual_encoder_relation_summary.csv",
]

SUBSETS_FILE_ORDER = [
    "multimodal_dual_encoder_emotion_subsets.csv",
    "multimodal_dual_encoder_irony_subsets.csv",
    "train_textual_relation_subsets.csv",
    "multimodal_dual_encoder_relation_subsets.csv",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Master Experiment Runner for MultiEMit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help="Comma-separated list of tasks to run: 'all', 'emotion', 'irony', 'relation'. Default: 'all'",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        help="Comma-separated list of stages to run: 'all', 'textual', 'multimodal'. Default: 'all'",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        default=False,
        help="Force retraining of checkpoints even if saved models exist in ./checkpoints/",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs for textual models. Default: 10",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training and feature extraction. Default: 16",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate for textual model fine-tuning. Default: 2e-5",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for training initialization and SGD heads (test splits stay locked to 42). Default: 42",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show scheduled commands and parameter plan without executing.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        default=False,
        help="Re-aggregate existing summary results in ./results/ without running any training.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        default=False,
        help="List available tasks, scripts, and current checkpoint status.",
    )
    return parser.parse_args()


def resolve_tasks_and_stages(tasks_arg: str, stages_arg: str):
    if tasks_arg.strip().lower() == "all":
        selected_tasks = ALL_TASKS
    else:
        selected_tasks = [t.strip().lower() for t in tasks_arg.split(",") if t.strip()]
        for t in selected_tasks:
            if t not in ALL_TASKS:
                raise ValueError(f"Unknown task '{t}'. Valid tasks: {ALL_TASKS}")

    if stages_arg.strip().lower() == "all":
        selected_stages = ALL_STAGES
    else:
        selected_stages = [s.strip().lower() for s in stages_arg.split(",") if s.strip()]
        for s in selected_stages:
            if s not in ALL_STAGES:
                raise ValueError(f"Unknown stage '{s}'. Valid stages: {ALL_STAGES}")

    plan = []
    for task in selected_tasks:
        for stage in selected_stages:
            script_file, label = TASK_SCRIPTS[task][stage]
            plan.append((task, stage, script_file, label))
    return plan


def print_status_list():
    print("\n" + "=" * 80)
    print("MultiEMit Task & Checkpoint Status")
    print("=" * 80)
    for task in ALL_TASKS:
        print(f"\n[{task.upper()}]")
        for stage in ALL_STAGES:
            script_file, label = TASK_SCRIPTS[task][stage]
            exists = (SCRIPT_DIR / script_file).exists()
            print(f"  • {stage.capitalize():<12} Script: {script_file} ({'FOUND' if exists else 'MISSING'})")

    print("\nCheckpoints in ./checkpoints/:")
    ckpts = sorted(CHECKPOINTS_DIR.glob("*"))
    if not ckpts:
        print("  (No checkpoints currently found)")
    else:
        for c in ckpts:
            if c.is_dir():
                has_cfg = (c / "config.json").exists()
                has_rcfg = (c / "run_config.json").exists()
                tag = "Complete HF Model + run_config.json" if (has_cfg and has_rcfg) else ("Complete HF Model" if has_cfg else "Incomplete")
                print(f"  • {c.name:<32} ({tag})")


def df_to_markdown_safe(df):
    try:
        return df.to_markdown(index=False)
    except Exception:
        cols = list(df.columns)
        rows = [cols, ["---"] * len(cols)] + [[str(val) for val in row] for row in df.values]
        col_widths = [max(len(str(r[i])) for r in rows) for i in range(len(cols))]
        lines = []
        for idx, r in enumerate(rows):
            if idx == 1:
                lines.append("| " + " | ".join("-" * w for w in col_widths) + " |")
            else:
                lines.append("| " + " | ".join(f"{r[i]:<{col_widths[i]}}" for i in range(len(cols))) + " |")
        return "\n".join(lines)


def aggregate_results(results_dir: Path):
    print("\n" + "=" * 80)
    print("AGGREGATING RESULTS...")
    print("=" * 80)

    dfs = []
    for csv_name in SUMMARY_FILE_ORDER:
        csv_path = results_dir / csv_name
        if csv_path.exists():
            try:
                dfs.append(pd.read_csv(csv_path))
            except Exception as e:
                print(f"Warning: could not read {csv_name}: {e}")

    if not dfs:
        print("No summary CSV files found in results/ to aggregate.")
        return

    combined_df = pd.concat(dfs, ignore_index=True)
    summary_csv = results_dir / "all_experiments_summary.csv"
    combined_df.to_csv(summary_csv, index=False)
    print(f"Unified summary saved to: {summary_csv}")

    subset_dfs = []
    for sub_name in SUBSETS_FILE_ORDER:
        sub_path = results_dir / sub_name
        if sub_path.exists():
            try:
                subset_dfs.append(pd.read_csv(sub_path))
            except Exception as e:
                print(f"Warning: could not read {sub_name}: {e}")

    combined_sub_df = pd.concat(subset_dfs, ignore_index=True) if subset_dfs else None
    if combined_sub_df is not None:
        subsets_csv = results_dir / "all_experiments_subsets.csv"
        combined_sub_df.to_csv(subsets_csv, index=False)
        print(f"Unified subsets summary saved to: {subsets_csv}")

    try:
        summary_xlsx = results_dir / "all_experiments_summary.xlsx"
        with pd.ExcelWriter(summary_xlsx) as writer:
            combined_df.to_excel(writer, sheet_name="Summary", index=False)
            if combined_sub_df is not None:
                combined_sub_df.to_excel(writer, sheet_name="Subsets_ADD_PAR_DIV", index=False)
        print(f"Unified summary Excel saved to: {summary_xlsx}")
    except Exception:
        pass

    # Markdown Report
    md_path = results_dir / "all_experiments_full_report.md"
    with open(md_path, "w", encoding="utf-8") as f_md:
        f_md.write("# MultiEMit Full Benchmark Experiments Report\n")
        f_md.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f_md.write("## 1. Unified Summary Metrics Table\n\n")
        f_md.write(df_to_markdown_safe(combined_df.round(4)))
        if combined_sub_df is not None:
            f_md.write("\n\n---\n\n## 2. Semiotic Relation Subsets Breakdown (ALL / ADD / PAR / DIV)\n\n")
            f_md.write(df_to_markdown_safe(combined_sub_df.round(4)))
        f_md.write("\n\n---\n\n## 3. Per-Task Classification Reports\n\n")

        report_files = [
            ("Textual Emotion Detection", "train_textual_emotion_reports.txt"),
            ("Multimodal Emotion Dual-Encoder", "multimodal_dual_encoder_emotion_reports.txt"),
            ("Textual Irony Detection", "train_textual_irony_reports.txt"),
            ("Multimodal Irony Dual-Encoder", "multimodal_dual_encoder_irony_reports.txt"),
            ("Textual Semiotic Relation Detection", "train_textual_relation_reports.txt"),
            ("Multimodal Relation Dual-Encoder", "multimodal_dual_encoder_relation_reports.txt"),
        ]

        for title, fname in report_files:
            rpath = results_dir / fname
            if rpath.exists():
                f_md.write(f"### {title}\n```\n")
                f_md.write(rpath.read_text(encoding="utf-8"))
                f_md.write("\n```\n\n")

    print(f"Full markdown report saved to: {md_path}")
    print("\n" + "=" * 80)
    print("SUMMARY RESULTS TABLE:")
    print("=" * 80)
    print(combined_df.round(4).to_string(index=False))


def main():
    args = parse_args()

    if args.list:
        print_status_list()
        return

    if args.aggregate_only:
        aggregate_results(RESULTS_DIR)
        return

    plan = resolve_tasks_and_stages(args.tasks, args.stages)

    print("\n" + "=" * 80)
    print("MultiEMit Master Experiment Runner")
    print(f"Tasks: {args.tasks} | Stages: {args.stages}")
    print(f"Parameters -> Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr} | Seed: {args.seed} | Force Retrain: {args.force_retrain}")
    print("=" * 80)
    print(f"Scheduled execution sequence ({len(plan)} scripts):")
    for i, (task, stage, script_file, label) in enumerate(plan, 1):
        print(f"  {i}. [{task.upper()} - {stage.upper()}] {script_file} — {label}")

    if args.dry_run:
        print("\n[DRY RUN] Execution skipped. Remove --dry-run to start training.")
        return

    python_bin = sys.executable
    total_start = time.time()
    success_count = 0

    for i, (task, stage, script_file, label) in enumerate(plan, 1):
        script_path = SCRIPT_DIR / script_file
        if not script_path.exists():
            print(f"\n[ERROR] Script {script_file} does not exist! Skipping.")
            continue

        cmd = [python_bin, str(script_path), "--seed", str(args.seed), "--batch-size", str(args.batch_size)]

        if stage == "textual":
            cmd.extend(["--epochs", str(args.epochs), "--lr", str(args.lr)])
            if args.force_retrain:
                cmd.append("--force-retrain")

        print("\n" + "=" * 80)
        print(f"[{i}/{len(plan)}] RUNNING: {label}")
        print(f"Command: {' '.join(cmd)}")
        print("=" * 80, flush=True)

        step_start = time.time()
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"

        try:
            subprocess.run(cmd, cwd=str(SCRIPT_DIR), env=env, check=True)
            elapsed = time.time() - step_start
            print(f"\n[SUCCESS] Completed {script_file} in {elapsed:.1f}s")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"\n[FAILED] Error running {script_file} (exit code {e.returncode})")
            choice = input("\nContinue with remaining experiments? [y/N]: ").strip().lower()
            if choice != "y":
                print("Aborting experiment run.")
                break

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"EXPERIMENT RUN FINISHED: {success_count}/{len(plan)} successful in {total_elapsed / 60:.2f} mins")
    print("=" * 80)

    if success_count > 0:
        aggregate_results(RESULTS_DIR)


if __name__ == "__main__":
    main()
