"""
aug_utils.py — Shared Utilities for Augmentation Pipeline
───────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import pandas as pd
import numpy as np

from config import SEED, SCORES_FILE

_REQUIRED_COLS = ["src_text", "tgt_text"]


# ─── Resume / completion helpers (used by step5a–5e) ─────────────────────────

def adapter_ready(path: str) -> bool:

    if not os.path.isdir(path):
        return False
    has_cfg = os.path.exists(os.path.join(path, "adapter_config.json"))
    has_wts = (os.path.exists(os.path.join(path, "adapter_model.bin")) or
               os.path.exists(os.path.join(path, "adapter_model.safetensors")))
    return has_cfg and has_wts


def load_all_scores() -> dict:
    """Load outputs/all_scores.json, returning {} if it doesn't exist yet."""
    if os.path.exists(SCORES_FILE):
        with open(SCORES_FILE) as f:
            return json.load(f)
    return {}


def experiment_done(pipeline_id: str, score_key: str,
                    adapter_path: str, pred_path: str) -> bool:

    if not adapter_ready(adapter_path):
        return False
    if not os.path.exists(pred_path):
        return False
    scores = load_all_scores()
    return f"{score_key}_{pipeline_id}" in scores


def needs_aug_generation(adapter_path: str, dataset_path: str) -> bool:

    return not (adapter_ready(adapter_path) and os.path.exists(dataset_path))


def combine_and_save(
    orig_df:    pd.DataFrame,
    aug_frames: list,
    save_path:  str,
    tag:        str = "",
) -> pd.DataFrame:

    orig = orig_df.copy()
    if "aug_type" not in orig.columns:
        orig["aug_type"] = "original"
    if "qa_score" not in orig.columns:
        orig["qa_score"] = np.nan

    frames = [orig]

    for i, aug in enumerate(aug_frames):
        if aug is None or len(aug) == 0:
            print(f"  [combine] aug_frame[{i}] is empty — skipping.")
            continue
        aug = aug.copy()
        frames.append(aug)

    combined = pd.concat(frames, ignore_index=True, sort=False)

    for col in _REQUIRED_COLS:
        if col not in combined.columns:
            raise ValueError(
                f"combine_and_save: required column '{col}' missing after merge. "
                f"Available: {list(combined.columns)}"
            )

    # Drop rows missing source or target
    before = len(combined)
    combined = combined.dropna(subset=_REQUIRED_COLS).reset_index(drop=True)
    if len(combined) < before:
        print(f"  [combine] Dropped {before-len(combined):,} rows with null src/tgt.")

    # Deduplicate on (src_text, tgt_text)
    before = len(combined)
    combined["_src_norm"] = combined["src_text"].astype(str).str.strip()
    combined["_tgt_norm"] = combined["tgt_text"].astype(str).str.strip()
    combined = (combined
                .drop_duplicates(subset=["_src_norm", "_tgt_norm"])
                .drop(columns=["_src_norm", "_tgt_norm"])
                .reset_index(drop=True))
    n_dupes = before - len(combined)
    if n_dupes:
        print(f"  [combine] Removed {n_dupes:,} duplicate (src, tgt) rows.")

    # Shuffle
    combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    combined.to_excel(save_path, index=False)

    # Summary
    aug_counts = combined["aug_type"].value_counts().to_dict() \
                 if "aug_type" in combined.columns else {}

    print(f"\n  [combine] [{tag}] Combined dataset:")
    print(f"    Total rows : {len(combined):,}")
    if aug_counts:
        for at, cnt in sorted(aug_counts.items()):
            print(f"    {at:<28}: {cnt:,}")
    print(f"    Saved → {save_path}")

    return combined


def dataset_stats(df: pd.DataFrame, label: str = "") -> dict:
    """Return and optionally print basic statistics."""
    stats = {"total": len(df)}
    if "aug_type" in df.columns:
        stats["by_aug_type"] = df["aug_type"].value_counts().to_dict()
    if label:
        print(f"\n  Dataset stats [{label}]: {stats}")
    return stats
