"""
Multimodal Emotion Detection — Dual-Encoder (FEEL-IT + ViT-224), Blind Text Head, and Direct Transformer Prediction.

Evaluated on MultiEMit Multimodal Test (mm_test = 129 posts, with ADD / PAR / DIV subset breakdown):
  1. Step 1 Base (emotion_textual_no_corr) — Raw text (No Correlations)
  2. Step 2 Raw Ablation (emotion_textual_step2_raw) — Raw text (No Correlations, A3 ablation)
  3. Step 2 Oracle (emotion_textual_corr) — Oracle Correlations without relation (matches Step 2 training schema, A5)
  4. Step 2 Oracle (emotion_textual_corr) — Oracle Correlations + semiotic relation (multimodal prompt)

Within each regime:
  - Fine-tuned model (direct prediction, threshold=0.40) [B2/B3]
  - Blind text (frozen 768-d [CLS] + ClassifierChain(SGD)) on raw and StandardScaler-normalized features [B1]
  - Dual-Encoder (frozen 768-d text [CLS] + 768-d ViT [CLS] + ClassifierChain(SGD)) on raw and StandardScaler features [B1]
"""

import argparse

from common_utils import (
    EMOTION_LABELS,
    add_multimodal_cli,
    run_multimodal_experiment,
    setup_console,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Emotion Dual-Encoder (FEEL-IT + ViT-224)")
    add_multimodal_cli(parser)
    return parser.parse_args()


def main():
    setup_console()
    args = parse_args()

    cfg = {
        "task": "emotion",
        "stem": "multimodal_dual_encoder_emotion",
        "multilabel": True,
        "labels": EMOTION_LABELS,
        "y_fn": lambda df: df[EMOTION_LABELS].values.astype(int),
        "direct_decode": "threshold",
        "regimes": [
            {
                "ckpt": "emotion_textual_no_corr",
                "label": "Step 1 Base (EMit A + ME)",
                "prefix": None,
                "prefix_name": "No",
                "direct": True,
            },
            {
                "ckpt": "emotion_textual_step2_raw",
                "label": "Step 2 Raw (Step 1 -> ME + MM)",
                "prefix": None,
                "prefix_name": "No",
                "direct": True,
            },
            {
                "ckpt": "emotion_textual_corr",
                "label": "Step 2 Oracle (Step 1 -> ME + MM)",
                "prefix": "oracle",
                "prefix_name": "Oracle (no rel)",
                "direct": True,
            },
            {
                "ckpt": "emotion_textual_corr",
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
