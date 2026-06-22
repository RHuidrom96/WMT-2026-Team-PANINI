"""
step5e_subword_reg.py — Subword Regularisation
─────────────────────────────────────────────────────────────────────────────
"""

import os
import argparse
import random
import numpy as np
import torch

from config import (
    ALL_PIPELINES, PIPELINE_CONFIGS, SEED,
    split_path, adapter_dir, aug_dataset_path, preds_path,
)
from aug_utils import experiment_done
from aug_finetune_eval import run_finetune_and_eval

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Exp name for checkpoints and scores
EXP_NAME = "exp5_subword_reg"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _adapter_ready(path: str) -> bool:
    """Return True only when a complete adapter (config + weights) is present."""
    if not os.path.isdir(path):
        return False
    has_cfg = os.path.exists(os.path.join(path, "adapter_config.json"))
    has_wts = (
        os.path.exists(os.path.join(path, "adapter_model.bin")) or
        os.path.exists(os.path.join(path, "adapter_model.safetensors"))
    )
    return has_cfg and has_wts


def _pick_start_adapter(pid: str) -> str:
    """Return the best available starting adapter for a pipeline (training path)."""
    for exp in ["exp3_bt_ft_iter", "exp2_bt_ft", "r1"]:
        path = adapter_dir(pid, exp)
        if _adapter_ready(path):
            print(f"  [SubwordReg] Starting from {exp} adapter: {path}")
            return path
    raise FileNotFoundError(
        f"No adapter found for pipeline {pid}. "
        f"Run at least step3_train_dora.py --pipeline {pid} first."
    )


def _pick_dataset(pid: str) -> str:
    """Return the best available augmented dataset for a pipeline (training path)."""
    for exp in ["exp3_bt_ft_iter", "exp2_bt_ft", "exp1_bt"]:
        path = aug_dataset_path(pid, exp)
        if os.path.exists(path):
            print(f"  [SubwordReg] Using dataset: {path}")
            return path
    path = split_path(pid, "train")
    print(f"  [SubwordReg] Falling back to base train split: {path}")
    return path


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pipeline", nargs="+", choices=ALL_PIPELINES, default=ALL_PIPELINES,
        metavar="ID",
        help="Pipeline(s) to process. Default: all.",
    )
    args = parser.parse_args()
    pipelines = args.pipeline

    print("\n" + "=" * 65)
    print("  STEP 5e — EXP-5: SUBWORD REGULARISATION (training-time)")
    print(f"  Pipelines : {pipelines}")
    print("=" * 65)

    dataset_files    = {}   # {pid: path | None}
    adapter_out_dirs = {}   # {pid: path}
    pred_files_map   = {}   # {pid: path}
    start_adapters   = {}   # {pid: path | None}
    eval_only_flags  = {}   # {pid: bool}  ← NEW: True = skip training entirely

    for pid in pipelines:
        cfg          = PIPELINE_CONFIGS[pid]
        exp5_adapter = adapter_dir(pid, EXP_NAME)
        exp5_preds   = preds_path(pid, EXP_NAME)

        adapter_out_dirs[pid] = exp5_adapter
        pred_files_map[pid]   = exp5_preds

        print(f"\n{'─' * 65}")
        print(f"  Pipeline {pid}  ({cfg['label']})")
        print(f"{'─' * 65}")

        # ── Case 1: Final adapter already present → eval-only ──────────────
        if _adapter_ready(exp5_adapter):
            print(f"  [MODE] Eval-only — final adapter found at:")
            print(f"         {exp5_adapter}")
            print(f"         Skipping all training and dataset loading.")
            eval_only_flags[pid] = True
            dataset_files[pid]   = None   # not needed
            start_adapters[pid]  = None   # not needed
            continue

        # ── Case 2: Scores already recorded too → fully done ───────────────
        if experiment_done(pid, "exp5", exp5_adapter, exp5_preds):
            print(f"  [MODE] Fully complete (adapter + preds + scores present).")
            print(f"         Nothing to do for pipeline {pid}.")
            eval_only_flags[pid] = True
            dataset_files[pid]   = None
            start_adapters[pid]  = None
            continue

        # ── Case 3: Need to train from scratch / resume ────────────────────
        print(f"  [MODE] Train + Eval")
        eval_only_flags[pid] = False
        start_adapters[pid]  = _pick_start_adapter(pid)
        dataset_files[pid]   = _pick_dataset(pid)

    # ── Decide whether any work needs doing at all ─────────────────────────
    pipelines_to_run = [pid for pid in pipelines
                        if not (eval_only_flags[pid] and
                                experiment_done(pid, "exp5",
                                                adapter_out_dirs[pid],
                                                pred_files_map[pid]))]

    if not pipelines_to_run:
        print(f"\n  All requested pipelines are fully complete. Nothing to run.")
        _print_summary(pipelines, adapter_out_dirs, pred_files_map, {})
        return

    # ── Fine-tune (skipped per-pipeline when eval_only) + Evaluate ─────────
    print(f"\n{'─' * 65}")
    print(f"  Fine-Tune & Evaluate — Exp-5 Subword Reg")
    print(f"  Pipelines to run : {pipelines_to_run}")
    print(f"  use_subword_reg  = True")
    print(f"{'─' * 65}")

    results = run_finetune_and_eval(
        score_key        = "exp5",
        dataset_files    = dataset_files,
        adapter_out_dirs = adapter_out_dirs,
        pred_files       = pred_files_map,
        pipelines        = pipelines_to_run,
        start_adapters   = start_adapters,
        label            = "Exp-5 (SubwordReg)",
        use_subword_reg  = True,
        eval_only_map    = eval_only_flags,   # ← passed through to runner
    )

    _print_summary(pipelines, adapter_out_dirs, pred_files_map, results)


def _print_summary(pipelines, adapter_out_dirs, pred_files_map, results):
    print(f"\n{'=' * 65}")
    print(f"  STEP 5e COMPLETE — Exp-5 (Subword Regularisation)")
    print(f"{'=' * 65}")
    for pid in pipelines:
        print(f"\n  Pipeline {pid}  ({PIPELINE_CONFIGS[pid]['label']})")
        print(f"    Exp-5 adapter : {adapter_out_dirs.get(pid, 'N/A')}")
        print(f"    Predictions   : {pred_files_map.get(pid, 'N/A')}")
        if pid in results:
            m = results[pid]
            print(f"    BLEU={m.get('BLEU', 0):.4f}  "
                  f"chrF={m.get('chrF', 0):.4f}  "
                  f"COMET={m.get('COMET', 0):.4f}")
    print(f"\n  Scores → outputs/all_scores.json  (keys: exp5_<pipeline>)")
    print()


if __name__ == "__main__":
    main()
