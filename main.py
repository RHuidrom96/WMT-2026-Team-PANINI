"""
main.py — End-to-End Pipeline 
─────────────────────────────────────────────────────────────────────────────

"""

import os
import sys
import argparse
import time
import subprocess
import traceback
from datetime import datetime

from config import (
    ALL_PIPELINES, SCORES_FILE, OUTPUT_DIR, FIGURES_DIR,
    split_path, adapter_dir, preds_path, aug_dataset_path,
)


# ─── Step registry ────────────────────────────────────────────────────────────

def _make_steps(pipelines: list) -> list:
    """Build the step list for the requested pipelines."""
    return [
        {
            "name":   "step1",
            "script": "step1_analysis.py",
            "args":   ["--pipeline"] + pipelines,
            "outputs": [split_path(p, sp)
                        for p in pipelines for sp in ["train", "dev", "test"]],
            "desc":   "Dataset cleaning & corpus analysis",
        },
        {
            "name":   "step2",
            "script": "step2_baseline_eval.py",
            "args":   ["--pipeline"] + pipelines,
            "outputs": [preds_path(p, "baseline") for p in pipelines] + [SCORES_FILE],
            "desc":   "Zero-shot baseline evaluation",
        },
        {
            "name":   "step3",
            "script": "step3_train_lora.py",
            "args":   ["--pipeline"] + pipelines,
            "outputs": [os.path.join(adapter_dir(p, "r1"), "adapter_config.json")
                        for p in pipelines],
            "desc":   "DoRA/LoRA fine-tuning (R1 adapters)",
        },
        {
            "name":   "step4",
            "script": "step4_finetuned_eval.py",
            "args":   ["--pipeline"] + pipelines,
            "outputs": [preds_path(p, "r1") for p in pipelines],
            "desc":   "R1 fine-tuned model evaluation",
        },
        {
            "name":   "step5a",
            "script": "step5a_aug_round1.py",
            "args":   ["--pipeline"] + pipelines,
            "outputs": [aug_dataset_path(p, "exp1_bt") for p in pipelines] +
                       [os.path.join(adapter_dir(p, "exp1_bt"), "adapter_config.json")
                        for p in pipelines] +
                       [preds_path(p, "exp1_bt") for p in pipelines],
            "desc":   "Exp-1: Back-Translation augmentation",
        },
        {
            "name":   "step5b",
            "script": "step5b_aug_round2.py",
            "args":   ["--pipeline"] + pipelines,
            "outputs": [aug_dataset_path(p, "exp2_bt_ft") for p in pipelines] +
                       [os.path.join(adapter_dir(p, "exp2_bt_ft"), "adapter_config.json")
                        for p in pipelines] +
                       [preds_path(p, "exp2_bt_ft") for p in pipelines],
            "desc":   "Exp-2: BT + Forward Translation",
        },
        {
            "name":   "step5c",
            "script": "step5c_aug_round3.py",
            "args":   ["--pipeline"] + pipelines,
            "outputs": [aug_dataset_path(p, "exp3_bt_ft_iter") for p in pipelines] +
                       [os.path.join(adapter_dir(p, "exp3_bt_ft_iter"), "adapter_config.json")
                        for p in pipelines] +
                       [preds_path(p, "exp3_bt_ft_iter") for p in pipelines],
            "desc":   "Exp-3: BT + FT + Iterative Retraining",
        },
        {
            "name":   "step5d",
            "script": "step5d_noise_injection.py",
            "args":   ["--pipeline"] + pipelines,
            "outputs": [aug_dataset_path(p, "exp4_noise") for p in pipelines] +
                       [os.path.join(adapter_dir(p, "exp4_noise"), "adapter_config.json")
                        for p in pipelines] +
                       [preds_path(p, "exp4_noise") for p in pipelines],
            "desc":   "Exp-4: Noise Injection augmentation",
        },
        {
            "name":   "step5e",
            "script": "step5e_subword_reg.py",
            "args":   ["--pipeline"] + pipelines,
            "outputs": [os.path.join(adapter_dir(p, "exp5_subword_reg"),
                                     "adapter_config.json")
                        for p in pipelines] +
                       [preds_path(p, "exp5_subword_reg") for p in pipelines],
            "desc":   "Exp-5: Subword Regularisation (training-time)",
        },
        {
            "name":   "step6",
            "script": "step6_compare.py",
            "args":   ["--pipeline"] + pipelines,
            "outputs": [
                os.path.join(OUTPUT_DIR, "comparison_table.csv"),
                os.path.join(OUTPUT_DIR, "results_latex.tex"),
                os.path.join(FIGURES_DIR, "all_pipelines_summary.png"),
            ],
            "desc":   "Comparison plots, significance tests, LaTeX table",
        },
    ]


VALID_STEP_NAMES = [
    "step1","step2","step3","step4",
    "step5a","step5b","step5c","step5d","step5e",
    "step6",
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def step_outputs_exist(step: dict) -> bool:
    return all(os.path.exists(p) for p in step["outputs"])


def missing_outputs(step: dict) -> list:
    return [p for p in step["outputs"] if not os.path.exists(p)]


def run_step(step: dict, extra_args: list = None) -> bool:
    cmd = [sys.executable, step["script"]] + step["args"] + (extra_args or [])
    print(f"\n  $ {' '.join(cmd)}")
    t0 = time.time()
    try:
        result  = subprocess.run(cmd, check=False)
        elapsed = time.time() - t0
        if result.returncode == 0:
            print(f"\n  ✓ {step['name']} completed in {elapsed/60:.1f} min")
            return True
        else:
            print(f"\n  ✗ {step['name']} FAILED (exit {result.returncode}) "
                  f"after {elapsed/60:.1f} min")
            return False
    except KeyboardInterrupt:
        elapsed = time.time() - t0
        print(f"\n  ⚠  {step['name']} interrupted after {elapsed/60:.1f} min")
        print(f"      Re-run main.py to resume from this step.")
        raise
    except Exception as e:
        print(f"\n  ✗ {step['name']} crashed: {e}")
        traceback.print_exc()
        return False


def banner(text: str, char: str = "█"):
    line = char * 72
    print(f"\n{line}\n  {text}\n{line}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pipeline", nargs="+", choices=ALL_PIPELINES, default=ALL_PIPELINES,
        metavar="ID",
        help="Pipeline(s) to process. Default: A B C D.",
    )
    parser.add_argument(
        "--start-from", choices=VALID_STEP_NAMES, default=None,
        help="Skip every step before this one.",
    )
    parser.add_argument(
        "--only", choices=VALID_STEP_NAMES, default=None,
        help="Run only this single step.",
    )
    parser.add_argument(
        "--steps", nargs="+", choices=VALID_STEP_NAMES, default=None,
        metavar="STEP",
        help="Run exactly these steps (in order).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run all steps even if outputs exist.",
    )
    parser.add_argument(
        "--force-step", choices=VALID_STEP_NAMES, default=None,
        help="Re-run only this step even if its outputs exist.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the execution plan without running anything.",
    )
    args = parser.parse_args()

    pipelines = args.pipeline
    all_steps = _make_steps(pipelines)

    banner("MANIPURI MT PIPELINE — IndicTrans2 1B + LoRA")
    print(f"  Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Pipelines  : {pipelines}")
    print(f"  Working dir: {os.getcwd()}")

    # ── Select which steps to attempt ─────────────────────────────────────────
    if args.only:
        planned = [s for s in all_steps if s["name"] == args.only]
    elif args.steps:
        step_map = {s["name"]: s for s in all_steps}
        planned  = [step_map[n] for n in args.steps if n in step_map]
    elif args.start_from:
        idx     = VALID_STEP_NAMES.index(args.start_from)
        names   = VALID_STEP_NAMES[idx:]
        planned = [s for s in all_steps if s["name"] in names]
    else:
        planned = all_steps

    # ── Print plan ─────────────────────────────────────────────────────────────
    print(f"\n  {'Step':<10} {'Status':<14} {'Description'}")
    print(f"  {'-'*74}")
    plan_rows = []
    for s in planned:
        forced = args.force or (args.force_step == s["name"])
        done   = step_outputs_exist(s) and not forced
        status = "FORCE" if forced else ("skip (done)" if done else "run")
        plan_rows.append((s, status))
        print(f"  {s['name']:<10} {status:<14} {s['desc']}")

    if args.dry_run:
        print(f"\n  --dry-run: stopping before execution.\n")
        return 0

    # ── Execute ────────────────────────────────────────────────────────────────
    overall_t0 = time.time()
    for step, status in plan_rows:
        if status == "skip (done)":
            print(f"\n  ⊙ {step['name']}: outputs present — skipping.")
            continue

        banner(f"RUNNING {step['name'].upper()} — {step['desc']}", char="═")

        extra = []
        if status == "FORCE" and step["name"] == "step3":
            extra = ["--force"]

        ok = run_step(step, extra_args=extra)
        if not ok:
            print(f"\n{'█'*72}")
            print(f"  PIPELINE HALTED at {step['name']}")
            print(f"{'█'*72}")
            print(f"\n  To resume: python main.py --pipeline {' '.join(pipelines)}"
                  f" --start-from {step['name']}")
            missing = missing_outputs(step)
            if missing:
                print(f"\n  Missing outputs:")
                for m in missing[:5]:
                    print(f"    - {m}")
            return 1

        missing = missing_outputs(step)
        if missing:
            print(f"\n  ⚠  {step['name']} returned 0 but outputs missing:")
            for m in missing[:5]:
                print(f"    - {m}")
            print("  Halting.")
            return 1

    elapsed = time.time() - overall_t0
    banner(f"PIPELINE COMPLETE  —  {elapsed/60:.1f} min total")
    print(f"\n  Pipelines processed : {pipelines}")
    print(f"  Scores              : {SCORES_FILE}")
    print(f"  Comparison table    : {os.path.join(OUTPUT_DIR, 'comparison_table.csv')}")
    print(f"  Figures             : {FIGURES_DIR}/")
    print(f"  LaTeX table         : {os.path.join(OUTPUT_DIR, 'results_latex.tex')}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
