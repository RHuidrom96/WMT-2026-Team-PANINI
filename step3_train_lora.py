"""
step3_train_dora.py — DoRA/LoRA Fine-tuning of IndicTrans2 1B  (resumable)
─────────────────────────────────────────────────────────────────────────────
Encoder-decoder fine-tuning with rsLoRA adapters..
"""

import os, gc, argparse, warnings, shutil, glob, re
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    EarlyStoppingCallback, set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType

warnings.filterwarnings("ignore")

from config import (
    ALL_PIPELINES, PIPELINE_CONFIGS,
    USE_DORA, USE_RSLORA,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_BIAS, LORA_TARGET_MODULES,
    NUM_TRAIN_EPOCHS, PER_DEVICE_TRAIN_BATCH, PER_DEVICE_EVAL_BATCH,
    GRAD_ACCUM_STEPS, LEARNING_RATE, LR_SCHEDULER, WARMUP_RATIO,
    WEIGHT_DECAY, MAX_GRAD_NORM, LABEL_SMOOTHING,
    SAVE_TOTAL_LIMIT, LOGGING_STEPS, EVAL_STEPS, SAVE_STEPS,
    EARLY_STOPPING_PATIENCE,
    BF16, FP16, MAX_SEQ_LEN, SEED,
    split_path, adapter_dir, get_base_model,
)

set_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if BF16 else (torch.float16 if FP16 else torch.float32)

try:
    from IndicTransToolkit.processor import IndicProcessor
except ImportError:
    raise ImportError(
        "IndicTransToolkit required. Install:\n"
        "  pip install git+https://github.com/VarunGumma/IndicTransToolkit.git"
    )


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def find_latest_checkpoint(out_dir: str):
    if not os.path.isdir(out_dir):
        return None
    ckpts = []
    for p in glob.glob(os.path.join(out_dir, "checkpoint-*")):
        m = re.match(r"checkpoint-(\d+)$", os.path.basename(p))
        if m and os.path.isdir(p):
            ckpts.append((int(m.group(1)), p))
    if not ckpts:
        return None
    ckpts.sort()
    return ckpts[-1][1]


def has_final_adapter(out_dir: str) -> bool:
    if not os.path.isdir(out_dir):
        return False
    cfg     = os.path.exists(os.path.join(out_dir, "adapter_config.json"))
    weights = (
        os.path.exists(os.path.join(out_dir, "adapter_model.bin")) or
        os.path.exists(os.path.join(out_dir, "adapter_model.safetensors"))
    )
    return cfg and weights


# ─── Dataset preparation ──────────────────────────────────────────────────────

def build_dataset(df: pd.DataFrame, pipeline_id: str,
                  tokenizer, ip: "IndicProcessor",
                  model_config) -> Dataset:
    cfg      = PIPELINE_CONFIGS[pipeline_id]
    src_lang = cfg["src_lang"]
    tgt_lang = cfg["tgt_lang"]

    df = df.dropna(subset=["src_text", "tgt_text"]).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"build_dataset: empty DataFrame for pipeline {pipeline_id}")

    sources = df["src_text"].astype(str).tolist()
    targets = df["tgt_text"].astype(str).tolist()

    sources_pp = ip.preprocess_batch(sources, src_lang=src_lang, tgt_lang=tgt_lang)
    targets_pp = ip.preprocess_batch(targets, src_lang=tgt_lang, tgt_lang=tgt_lang)

    ds = Dataset.from_dict({"src": sources_pp, "tgt": targets_pp})

    decoder_start = model_config.decoder_start_token_id
    pad_id        = model_config.pad_token_id
    if decoder_start is None:
        raise ValueError("model.config.decoder_start_token_id is None")

    def shift_right(ids):
        return [decoder_start] + ids[:-1]

    def tokenize_fn(batch):
        enc = tokenizer(batch["src"], truncation=True, padding=False,
                        max_length=MAX_SEQ_LEN)
        lbl = tokenizer(text_target=batch["tgt"], truncation=True,
                        padding=False, max_length=MAX_SEQ_LEN)
        enc["labels"]            = lbl["input_ids"]
        enc["decoder_input_ids"] = [shift_right(s) for s in lbl["input_ids"]]
        return enc

    return ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names,
                  desc=f"Tokenizing pipeline {pipeline_id}")


def make_collator(tokenizer):
    pad_id = tokenizer.pad_token_id

    def collator(features):
        def r8(n): return ((n + 7) // 8) * 8
        input_max = r8(max(len(f["input_ids"])         for f in features))
        label_max = r8(max(len(f["labels"])            for f in features))
        dec_max   = r8(max(len(f["decoder_input_ids"]) for f in features))

        batch = {"input_ids": [], "attention_mask": [],
                 "labels": [], "decoder_input_ids": []}
        for f in features:
            iids  = f["input_ids"]
            amask = f.get("attention_mask", [1]*len(iids))
            lbls  = f["labels"]
            dids  = f["decoder_input_ids"]
            batch["input_ids"].append(        iids  + [pad_id]*( input_max - len(iids)))
            batch["attention_mask"].append(   amask + [0]      *(input_max - len(amask)))
            batch["labels"].append(           lbls  + [-100]   *(label_max - len(lbls)))
            batch["decoder_input_ids"].append(dids  + [pad_id]*(dec_max   - len(dids)))
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}

    return collator


# ─── Loss curve ───────────────────────────────────────────────────────────────

def _plot_loss(history, save_path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_pts = [(h["epoch"], h["loss"])      for h in history if "loss"      in h]
    eval_pts  = [(h["epoch"], h["eval_loss"]) for h in history if "eval_loss" in h]

    fig, ax = plt.subplots(figsize=(9, 5))
    if train_pts:
        ax.plot(*zip(*train_pts), label="Train Loss", color="#4C72B0", lw=2)
    if eval_pts:
        ax.plot(*zip(*eval_pts),  label="Val Loss",   color="#DD8452", lw=2, ls="--")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend()
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  ✓ Loss curve → {save_path}")


# ─── Training ─────────────────────────────────────────────────────────────────

def train_pipeline(pipeline_id: str, force: bool = False):
    cfg        = PIPELINE_CONFIGS[pipeline_id]
    out_dir    = adapter_dir(pipeline_id, "r1")
    model_name = get_base_model(pipeline_id)

    print(f"\n{'='*64}")
    print(f"  TRAINING  Pipeline {pipeline_id}  —  {cfg['label']}")
    print(f"  Base model : {model_name}")
    print(f"  Adapter    → {out_dir}")
    print(f"{'='*64}")

    if has_final_adapter(out_dir) and not force:
        print(f"  [SKIP] Final adapter already exists at {out_dir}.")
        return

    if force:
        for ckpt in glob.glob(os.path.join(out_dir, "checkpoint-*")):
            shutil.rmtree(ckpt, ignore_errors=True)
        for fn in ["adapter_config.json", "adapter_model.bin",
                   "adapter_model.safetensors", "tokenizer_config.json"]:
            fp = os.path.join(out_dir, fn)
            if os.path.exists(fp):
                os.remove(fp)
        print("  [FORCE] Cleared existing checkpoints/adapter files.")

    resume_ckpt = find_latest_checkpoint(out_dir)
    if resume_ckpt:
        print(f"  [RESUME] Resuming from {resume_ckpt}")
    else:
        print(f"  [TRAIN] Starting fresh.")

    train_df = pd.read_excel(split_path(pipeline_id, "train"))
    dev_df   = pd.read_excel(split_path(pipeline_id, "dev"))
    print(f"  Train: {len(train_df):,}  |  Dev: {len(dev_df):,}")

    tokenizer    = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model_kwargs = {"trust_remote_code": True, "torch_dtype": DTYPE}
    try:
        import flash_attn  # noqa: F401
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("  flash_attn : enabled")
    except ImportError:
        pass

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)

    ip    = IndicProcessor(inference=False)

    print("  Building datasets ...")
    train_ds = build_dataset(train_df, pipeline_id, tokenizer, ip, model.config)
    eval_ds  = build_dataset(dev_df,   pipeline_id, tokenizer, ip, model.config)
    print(f"  Train DS: {len(train_ds):,}  |  Dev DS: {len(eval_ds):,}")

    # ── LoRA / DoRA config ────────────────────────────────────────────────────
    peft_cfg = LoraConfig(
        task_type         = TaskType.SEQ_2_SEQ_LM,
        r                 = LORA_R,
        lora_alpha        = LORA_ALPHA,
        lora_dropout      = LORA_DROPOUT,
        bias              = LORA_BIAS,
        target_modules    = LORA_TARGET_MODULES,
        use_dora          = USE_DORA,
        use_rslora        = USE_RSLORA,
    )

    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    model = model.to(DEVICE)

    collator = make_collator(tokenizer)

    adapter_type = "DoRA" if USE_DORA else "LoRA"
    print(f"  Adapter type : {adapter_type}  r={LORA_R}  alpha={LORA_ALPHA}")

    training_args = Seq2SeqTrainingArguments(
        output_dir                  = out_dir,
        overwrite_output_dir        = False,
        num_train_epochs            = NUM_TRAIN_EPOCHS,
        per_device_train_batch_size = PER_DEVICE_TRAIN_BATCH,
        per_device_eval_batch_size  = PER_DEVICE_EVAL_BATCH,
        gradient_accumulation_steps = GRAD_ACCUM_STEPS,
        learning_rate               = LEARNING_RATE,
        lr_scheduler_type           = LR_SCHEDULER,
        warmup_ratio                = WARMUP_RATIO,
        weight_decay                = WEIGHT_DECAY,
        max_grad_norm               = MAX_GRAD_NORM,
        label_smoothing_factor      = LABEL_SMOOTHING,
        bf16                        = BF16,
        fp16                        = FP16,
        eval_strategy               = "steps",
        eval_steps                  = EVAL_STEPS,
        save_strategy               = "steps",
        save_steps                  = SAVE_STEPS,
        save_total_limit            = SAVE_TOTAL_LIMIT,
        logging_steps               = LOGGING_STEPS,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        predict_with_generate       = False,
        gradient_checkpointing      = True,
        report_to                   = "none",
        seed                        = SEED,
        dataloader_num_workers      = 2,
        remove_unused_columns       = False,
        group_by_length             = True,
    )

    trainer = Seq2SeqTrainer(
        model         = model,
        args          = training_args,
        train_dataset = train_ds,
        eval_dataset  = eval_ds,
        data_collator = collator,
        tokenizer     = tokenizer,
        callbacks     = [EarlyStoppingCallback(
                            early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    trainer.train(resume_from_checkpoint=resume_ckpt)

    print(f"\n  Saving final adapter → {out_dir}")
    trainer.model.save_pretrained(out_dir)

    print(f"  ✓ Saved pipeline {pipeline_id} R1 adapter")

    from config import FIGURES_DIR
    fig_dir = os.path.join(FIGURES_DIR, f"pipeline_{pipeline_id}")
    os.makedirs(fig_dir, exist_ok=True)
    _plot_loss(
        trainer.state.log_history,
        os.path.join(fig_dir, "loss_r1.png"),
        f"Pipeline {pipeline_id} R1 Training Loss",
    )

    del model, trainer, train_ds, eval_ds
    gc.collect()
    torch.cuda.empty_cache()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--pipeline", nargs="+", choices=ALL_PIPELINES, default=ALL_PIPELINES,
        metavar="ID",
    )
    parser.add_argument("--force", action="store_true",
                        help="Delete existing checkpoints and retrain from scratch.")
    args = parser.parse_args()

    print("\n" + "="*64)
    print("  STEP 3 — DoRA/LoRA FINE-TUNING (R1 adapters)")
    print(f"  Pipelines : {args.pipeline}  |  Force : {args.force}")
    print("="*64)

    for pid in args.pipeline:
        train_pipeline(pid, force=args.force)

    print("\n" + "="*64)
    print("  STEP 3 COMPLETE")
    print("="*64)
    for pid in args.pipeline:
        print(f"  Pipeline {pid} R1 adapter → {adapter_dir(pid, 'r1')}")
    print()


if __name__ == "__main__":
    main()
