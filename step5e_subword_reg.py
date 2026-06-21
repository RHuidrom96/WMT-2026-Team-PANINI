"""
step5e_subword_reg.py — Subword Regularisation
─────────────────────────────────────────────────────────────────────────────
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
import torch

from config import (
    ALL_PIPELINES, PIPELINE_CONFIGS, SEED,
    split_path, adapter_dir, aug_dataset_path, preds_path,
    OUTPUT_DIR,
)
from aug_utils import dataset_stats, experiment_done
from aug_finetune_eval import run_finetune_and_eval

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Exp name for checkpoints and scores
EXP_NAME = "exp5_subword_reg"


def _adapter_ready(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    has_cfg = os.path.exists(os.path.join(path, "adapter_config.json"))
    has_wts = (os.path.exists(os.path.join(path, "adapter_model.bin")) or
               os.path.exists(os.path.join(path, "adapter_model.safetensors")))
    return has_cfg and has_wts


def _pick_start_adapter(pid: str) -> str:

    for exp in ["exp3_bt_ft_iter", "exp2_bt_ft", "r1"]:
        path = adapter_dir(pid, exp)
        if _adapter_ready(path):
            print(f"  [SubwordReg] Starting from {exp} adapter: {path}")
            return path
    raise FileNotFoundError(
        f"No adapter found for pipeline {pid}. "
        f"Run at least step3_train_lora.py --pipeline {pid} first."
    )


def _pick_dataset(pid: str) -> str:

    for exp in ["exp3_bt_ft_iter", "exp2_bt_ft", "exp1_bt"]:
        path = aug_dataset_path(pid, exp)
        if os.path.exists(path):
            print(f"  [SubwordReg] Using dataset: {path}")
            return path
    path = split_path(pid, "train")
    print(f"  [SubwordReg] Falling back to base train: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--pipeline", nargs="+", choices=ALL_PIPELINES, default=ALL_PIPELINES,
        metavar="ID",
    )
    args = parser.parse_args()
    pipelines = args.pipeline

    print("\n" + "=" * 65)
    print("  STEP 5e — EXP-5: SUBWORD REGULARISATION (training-time)")
    print(f"  Pipelines : {pipelines}")
    print("=" * 65)

    dataset_files    = {}
    adapter_out_dirs = {}
    pred_files_map   = {}
    start_adapters   = {}

    for pid in pipelines:
        cfg = PIPELINE_CONFIGS[pid]
        print(f"\n{'─' * 65}")
        print(f"  Pipeline {pid}  ({cfg['label']})")
        print(f"{'─' * 65}")

        exp5_adapter = adapter_dir(pid, EXP_NAME)
        exp5_preds   = preds_path(pid, EXP_NAME)
        adapter_out_dirs[pid] = exp5_adapter
        pred_files_map[pid]   = exp5_preds

        # ── Resume check: Exp-5 fully complete already? ─────────────────────────
        if experiment_done(pid, "exp5", exp5_adapter, exp5_preds):
            print(f"  [RESUME] Exp-5 already fully complete "
                  f"(adapter + preds + scores present).")
            print(f"           Skipping pipeline {pid} entirely.")
            dataset_files[pid]  = _pick_dataset(pid)
            start_adapters[pid] = exp5_adapter
            continue

        # ── Resume check: adapter trained, eval still pending? ────────────
        if _adapter_ready(exp5_adapter):
            print(f"  [RESUME] Exp-5 adapter already trained for pipeline {pid}; "
                  f"will (re)run evaluation only.")
            dataset_files[pid]  = _pick_dataset(pid)
            start_adapters[pid] = exp5_adapter   # unused — training is skipped
            df = pd.read_excel(dataset_files[pid])
            df = df.dropna(subset=["src_text", "tgt_text"]).reset_index(drop=True)
            print(f"  Dataset rows: {len(df):,}")
            dataset_stats(df, f"pipeline {pid} subword_reg input")
            continue

        # Pick the best starting adapter and dataset
        start_adapters[pid]   = _pick_start_adapter(pid)
        dataset_files[pid]    = _pick_dataset(pid)

        # Log dataset stats
        df = pd.read_excel(dataset_files[pid])
        df = df.dropna(subset=["src_text", "tgt_text"]).reset_index(drop=True)
        print(f"  Dataset rows: {len(df):,}")
        dataset_stats(df, f"pipeline {pid} subword_reg input")

    # ── Fine-tune + evaluate with subword regularisation enabled ──────────────
    print(f"\n{'─' * 65}")
    print(f"  Fine-Tune & Evaluate — Exp-5 Subword Reg  ({pipelines})")
    print(f"  use_subword_reg = True")
    print(f"{'─' * 65}")

    results = run_finetune_and_eval(
        score_key        = "exp5",
        dataset_files    = dataset_files,
        adapter_out_dirs = adapter_out_dirs,
        pred_files       = pred_files_map,
        pipelines        = pipelines,
        start_adapters   = start_adapters,
        label            = "Exp-5 (SubwordReg)",
        use_subword_reg  = True,   # ← enables SentencePiece sampling in collator
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  STEP 5e COMPLETE — Exp-5 (Subword Regularisation)")
    print(f"{'=' * 65}")
    for pid in pipelines:
        print(f"\n  Pipeline {pid}  ({PIPELINE_CONFIGS[pid]['label']})")
        print(f"    Start adapter : {start_adapters.get(pid, 'N/A')}")
        print(f"    Exp-5 adapter : {adapter_dir(pid, EXP_NAME)}")
        print(f"    Dataset       : {dataset_files.get(pid, 'N/A')}")
        print(f"    Preds         : {pred_files_map.get(pid, 'N/A')}")
        if pid in results:
            m = results[pid]
            print(f"    BLEU={m.get('BLEU',0):.4f}  chrF={m.get('chrF',0):.4f}  "
                  f"COMET={m.get('COMET',0):.4f}")
    print(f"\n  Scores → outputs/all_scores.json  (keys: exp5_<pipeline>)")
    print()


if __name__ == "__main__":
    main()
