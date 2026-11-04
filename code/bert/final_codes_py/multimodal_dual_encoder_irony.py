"""
Multimodal Irony & Sarcasm Detection — Dual-Encoder (UmBERTo + ViT-224), Blind Text Head, and Direct Transformer Prediction.

Evaluated on MultiEMit Multimodal Test (mm_test = 129 posts, with ADD / PAR / DIV subset breakdown):
  1. Step 1 Base (irony_textual_no_corr) — Raw text (No Correlations)
  2. Step 2 Raw Ablation (irony_textual_step2_raw) — Raw text (No Correlations, A3 ablation)
  3. Step 2 Oracle (irony_textual_corr) — Oracle Correlations without relation (matches Step 2 training schema, A5)
  4. Step 2 Oracle (irony_textual_corr) — Oracle Correlations + semiotic relation (multimodal prompt)

Within each regime:
  - Fine-tuned model (direct prediction, argmax) [B2/B3]
  - Blind text (frozen 768-d [CLS] + SGDClassifier) on raw and StandardScaler-normalized features [B1]
  - Dual-Encoder (frozen 768-d text [CLS] + 768-d ViT [CLS] + SGDClassifier) on raw and StandardScaler features [B1]
"""

import argparse

from common_utils import (
    IRONY_LABELS,
    add_multimodal_cli,
    row_to_irony_class,
    run_multimodal_experiment,
    setup_console,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Irony Dual-Encoder (UmBERTo + ViT-224)")
    add_multimodal_cli(parser)
    return parser.parse_args()


def main():
    setup_console()
    args = parse_args()

    cfg = {
        "task": "irony",
        "stem": "multimodal_dual_encoder_irony",
        "multilabel": False,
        "labels": IRONY_LABELS,
        "y_fn": lambda df: df.apply(row_to_irony_class, axis=1).values.astype(int),
        "direct_decode": "argmax",
        "regimes": [
            {
                "ckpt": "irony_textual_no_corr",
                "label": "Step 1 Base (IronITA + ME)",
                "prefix": None,
                "prefix_name": "No",
                "direct": True,
            },
            {
                "ckpt": "irony_textual_step2_raw",
                "label": "Step 2 Raw (Step 1 -> ME + MM)",
                "prefix": None,
                "prefix_name": "No",
                "direct": True,
            },
            {
                "ckpt": "irony_textual_corr",
                "label": "Step 2 Oracle (Step 1 -> ME + MM)",
                "prefix": "oracle",
                "prefix_name": "Oracle (no rel)",
                "direct": True,
            },
            {
                "ckpt": "irony_textual_corr",
                "label": "Step 2 Oracle (Step 1 -> ME + MM)",
                "prefix": "oracle_rel",
                "prefix_name": "Oracle (+ rel)",
                "direct": True,
            },
        ],
    }
    run_multimodal_experiment(cfg, args)


if __name__ == "__main__":
    main()
