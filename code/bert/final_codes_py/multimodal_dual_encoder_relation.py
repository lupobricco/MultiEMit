"""
Multimodal Semiotic Relation Classification — Dual-Encoder (UmBERTo + ViT-224), Blind Text Head, and Direct Prediction.

Evaluated on MultiEMit Multimodal Test (mm_test = 129 posts, with ADD / PAR / DIV subset breakdown):
  1. Stage A Raw (relation_textual_no_corr) — Raw text (No Correlations)
  2. Stage B Oracle (relation_textual_corr) — Oracle Correlations (emotions + target + irony/sarcasm + offensiveness)

Within each regime:
  - Fine-tuned model (direct prediction, argmax) [B2/B3]
  - Blind text (frozen 768-d [CLS] + SGDClassifier) on raw and StandardScaler-normalized features [B1]
  - Dual-Encoder (frozen 768-d text [CLS] + 768-d ViT [CLS] + SGDClassifier) on raw and StandardScaler features [B1]
"""

import argparse

from common_utils import (
    RELATION_LABELS,
    add_multimodal_cli,
    row_to_relation_class,
    run_multimodal_experiment,
    setup_console,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Semiotic Relation Dual-Encoder (UmBERTo + ViT-224)")
    add_multimodal_cli(parser)
    return parser.parse_args()


def main():
    setup_console()
    args = parse_args()

    cfg = {
        "task": "relation",
        "stem": "multimodal_dual_encoder_relation",
        "multilabel": False,
        "labels": RELATION_LABELS,
        "y_fn": lambda df: df.apply(row_to_relation_class, axis=1).values.astype(int),
        "direct_decode": "argmax",
        "regimes": [
            {
                "ckpt": "relation_textual_no_corr",
                "label": "Stage A: MM Train Raw",
                "prefix": None,
                "prefix_name": "No",
                "direct": True,
            },
            {
                "ckpt": "relation_textual_corr",
                "label": "Stage B: MM Train Oracle",
                "prefix": "oracle",
                "prefix_name": "Oracle",
                "direct": True,
            },
        ],
    }
    run_multimodal_experiment(cfg, args)


if __name__ == "__main__":
    main()
