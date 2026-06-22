"""
aug_finetune_eval.py — Shared Fine-Tune + Eval Runner (Direction-Specific)
"""

import os, json, glob, re, gc, warnings
import numpy as np
import pandas as pd
import torch
import unicodedata
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    EarlyStoppingCallback, set_seed,
)
from peft import PeftModel
import sacrebleu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

try:
    from IndicTransToolkit.processor import IndicProcessor
except ImportError:
    raise ImportError(
        "IndicTransToolkit required.\n"
        "  pip install git+https://github.com/VarunGumma/IndicTransToolkit.git"
    )

from config import (
    ALL_PIPELINES, PIPELINE_CONFIGS,
    OUTPUT_DIR, FIGURES_DIR, SCORES_FILE,
    SEED, MAX_SEQ_LEN, BF16, FP16,
    USE_DORA, LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_BIAS, LORA_TARGET_MODULES,
    NUM_TRAIN_EPOCHS_AUG, PER_DEVICE_TRAIN_BATCH, PER_DEVICE_EVAL_BATCH,
    GRAD_ACCUM_STEPS, LEARNING_RATE_AUG, LR_SCHEDULER, WARMUP_RATIO,
    WEIGHT_DECAY, MAX_GRAD_NORM, LABEL_SMOOTHING,
    SAVE_TOTAL_LIMIT, LOGGING_STEPS, EVAL_STEPS, SAVE_STEPS,
    EARLY_STOPPING_PATIENCE,
    INFER_BATCH_SIZE, NUM_BEAMS, MAX_NEW_TOKENS,
    LENGTH_PENALTY, NO_REPEAT_NGRAM_SIZE,
    COMET_MODEL, HALL_LEN_RATIO_LOW, HALL_LEN_RATIO_HIGH, HALL_COPY_RATIO,
    SYSTEM_COLORS, SUBWORD_REG_ALPHA,
    split_path, adapter_dir, get_base_model, sacrebleu_tokenize,
)

set_seed(SEED)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if BF16 else (torch.float16 if FP16 else torch.float32)

_LANG_TAG_RE = re.compile(r'^(?:[a-z]{3}_[A-Za-z]{4}\s+){1,2}')


def _strip_lang_tag(text: str) -> str:
    return _LANG_TAG_RE.sub("", text).strip()

def _strip_lang_tags(texts: list) -> list:
    return [unicodedata.normalize("NFC", _strip_lang_tag(t)) for t in texts]


# ─── Output-side script normalization (reuses step1_analysis.py) ─────────────

try:
    from step1_analysis import (
        _apply_script_normalization,
        _strip_ref_tags,
        _clean_text,
    )
except ImportError:
    _apply_script_normalization = None
    _strip_ref_tags = None
    _clean_text = None
    warnings.warn(
        "Could not import normalization helpers from step1_analysis.py "
        "(file not found on path). Falling back to no-op — predictions will "
        "NOT have digits/script/Apun-Iyek-spacing normalized or ref-tags "
    )

def _normalize_output_script(texts: list, tgt_lang: str) -> list:
    if _apply_script_normalization is None:
        return texts
    texts = _apply_script_normalization(pd.Series(texts), tgt_lang).tolist()
    texts = [_strip_ref_tags(t) for t in texts]
    texts = [_clean_text(t) for t in texts]
    return texts


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def _has_final_adapter(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    cfg     = os.path.exists(os.path.join(path, "adapter_config.json"))
    weights = (os.path.exists(os.path.join(path, "adapter_model.bin")) or
               os.path.exists(os.path.join(path, "adapter_model.safetensors")))
    return cfg and weights


def _find_latest_checkpoint(out_dir: str):
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
    latest = ckpts[-1][1]
    print(f"  [RESUME] Checkpoint found: {latest}")
    return latest


# ─── Scores persistence ───────────────────────────────────────────────────────

def _load_scores() -> dict:
    if os.path.exists(SCORES_FILE):
        with open(SCORES_FILE) as f:
            return json.load(f)
    return {}

def _save_scores(scores: dict):
    with open(SCORES_FILE, "w") as f:
        json.dump(scores, f, indent=2)


# ─── Dataset preparation ──────────────────────────────────────────────────────

def _build_dataset(df: pd.DataFrame, pipeline_id: str, tokenizer,
                   ip: "IndicProcessor", model_config) -> Dataset:
    cfg      = PIPELINE_CONFIGS[pipeline_id]
    src_lang = cfg["src_lang"]
    tgt_lang = cfg["tgt_lang"]

    df = df.dropna(subset=["src_text", "tgt_text"]).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(f"_build_dataset: no rows for pipeline {pipeline_id}")

    sources = df["src_text"].astype(str).tolist()
    targets = df["tgt_text"].astype(str).tolist()

    sources_pp = ip.preprocess_batch(sources, src_lang=src_lang, tgt_lang=tgt_lang)
    targets_pp = ip.preprocess_batch(targets, src_lang=tgt_lang, tgt_lang=tgt_lang)

    ds = Dataset.from_dict({"src": sources_pp, "tgt": targets_pp})

    decoder_start = model_config.decoder_start_token_id
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


def _make_collator(tokenizer):
    pad_id = tokenizer.pad_token_id

    def collator(features):
        def r8(n): return ((n+7)//8)*8
        input_max = r8(max(len(f["input_ids"])         for f in features))
        label_max = r8(max(len(f["labels"])            for f in features))
        dec_max   = r8(max(len(f["decoder_input_ids"]) for f in features))
        batch = {"input_ids":[], "attention_mask":[], "labels":[], "decoder_input_ids":[]}
        for f in features:
            iids  = f["input_ids"]
            amask = f.get("attention_mask", [1]*len(iids))
            lbls  = f["labels"]
            dids  = f["decoder_input_ids"]
            batch["input_ids"].append(        iids  + [pad_id]*(input_max-len(iids)))
            batch["attention_mask"].append(   amask + [0]     *(input_max-len(amask)))
            batch["labels"].append(           lbls  + [-100]  *(label_max-len(lbls)))
            batch["decoder_input_ids"].append(dids  + [pad_id]*(dec_max  -len(dids)))
        return {k: torch.tensor(v, dtype=torch.long) for k,v in batch.items()}

    return collator


# ─── Inference ────────────────────────────────────────────────────────────────

def _translate_it2(sentences, model, tokenizer, ip, src_lang, tgt_lang):
    batch  = ip.preprocess_batch(sentences, src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tokenizer(batch, truncation=True, padding="longest",
                       return_tensors="pt", max_length=MAX_SEQ_LEN).to(DEVICE)
    with torch.no_grad():
        generated = model.generate(
            **inputs, use_cache=True, min_length=0,
            max_new_tokens=MAX_NEW_TOKENS, num_beams=NUM_BEAMS,
            length_penalty=LENGTH_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE, early_stopping=True,
        )
    with tokenizer.as_target_tokenizer():
        decoded = tokenizer.batch_decode(
            generated.detach().cpu().tolist(),
            skip_special_tokens=True, clean_up_tokenization_spaces=True,
        )
    out = _strip_lang_tags(ip.postprocess_batch(decoded, lang=tgt_lang))
    return _normalize_output_script(out, tgt_lang)


def _run_inference(sources, model, tokenizer, ip, src_lang, tgt_lang, desc=""):
    all_preds = []
    for i in tqdm(range(0, len(sources), INFER_BATCH_SIZE),
                  desc=f"  Translating [{desc}]"):
        chunk = sources[i:i+INFER_BATCH_SIZE]
        all_preds.extend(_translate_it2(chunk, model, tokenizer, ip, src_lang, tgt_lang))
    return all_preds


# ─── Metrics ──────────────────────────────────────────────────────────────────

def _compute_sacre(preds, refs, pipeline_id):
    refs  = [unicodedata.normalize("NFC", r) for r in refs]
    preds = [unicodedata.normalize("NFC", p) for p in preds]
    tok   = sacrebleu_tokenize(pipeline_id)
    rw    = [refs]
    return {
        "BLEU":   round(sacrebleu.corpus_bleu(preds, rw, tokenize=tok).score, 4),
        "chrF":   round(sacrebleu.corpus_chrf(preds, rw).score, 4),
        "chrF++": round(sacrebleu.corpus_chrf(preds, rw, word_order=2).score, 4),
        "TER":    round(sacrebleu.corpus_ter(preds, rw).score, 4),
    }


def _compute_comet(srcs, preds, refs):
    try:
        from comet import download_model, load_from_checkpoint
        comet_model = load_from_checkpoint(download_model(COMET_MODEL))
        data  = [{"src":s,"mt":p,"ref":r} for s,p,r in zip(srcs,preds,refs)]
        out   = comet_model.predict(data, batch_size=16,
                                    gpus=1 if DEVICE=="cuda" else 0)
        score = round(float(out.system_score)*100, 4)
        del comet_model; gc.collect(); torch.cuda.empty_cache()
        return score
    except Exception as e:
        print(f"  [COMET] Skipped — {e}")
        return None


# ─── Error & hallucination analysis ──────────────────────────────────────────

def _error_analysis(preds, refs, score_key, pipeline_id, top_n=5):
    sent_bleus = []
    for p, r in zip(preds, refs):
        try:
            s = sacrebleu.sentence_bleu(p, [r]).score
        except Exception:
            s = 0.0
        sent_bleus.append(s)
    sb = np.array(sent_bleus)
    print(f"\n  ERROR ANALYSIS [{score_key}] [pipeline {pipeline_id}]")
    print(f"  Sent-BLEU  mean={sb.mean():.2f}  median={np.median(sb):.2f}  "
          f"std={sb.std():.2f}")
    print(f"  % BLEU=0   : {(sb==0).mean()*100:.1f}%  "
          f"% BLEU≥30  : {(sb>=30).mean()*100:.1f}%")
    worst = np.argsort(sb)[:top_n]
    print(f"  Worst {top_n}:")
    for rank, i in enumerate(worst, 1):
        print(f"  [{rank}] BLEU={sb[i]:.2f}  REF:{refs[i][:80]}  PRED:{preds[i][:80]}")

    fig_dir = os.path.join(FIGURES_DIR, f"pipeline_{pipeline_id}")
    os.makedirs(fig_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9,5))
    ax.hist(sb, bins=40, color=SYSTEM_COLORS.get(score_key,"#888"),
            edgecolor="white", alpha=0.85)
    ax.axvline(sb.mean(), color="red", linestyle="--", lw=1.5,
               label=f"Mean={sb.mean():.2f}")
    ax.set_title(f"Sent-BLEU [{score_key}] [pipeline {pipeline_id}]",
                 fontweight="bold")
    ax.set_xlabel("Sentence BLEU"); ax.set_ylabel("Frequency"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"err_{score_key}_sent_bleu.png"))
    plt.close()


def _hallucination_analysis(srcs, preds, refs, score_key, pipeline_id):
    n = len(preds)
    omission = repetition = copy_hall = 0
    rows = []
    for i, (src, pred, ref) in enumerate(zip(srcs, preds, refs)):
        pt    = pred.split() if pred.strip() else [""]
        rt    = ref.split()  if ref.strip()  else [""]
        st    = src.split()  if src.strip()  else [""]
        ratio = len(pt) / max(len(rt), 1)
        flags = []
        if ratio < HALL_LEN_RATIO_LOW:
            omission += 1; flags.append("omission")
        if ratio > HALL_LEN_RATIO_HIGH:
            repetition += 1; flags.append("repetition")
        ps = set(t.lower() for t in pt)
        ss = set(t.lower() for t in st)
        if ss and len(ps & ss)/max(len(ps),1) > HALL_COPY_RATIO:
            copy_hall += 1; flags.append("copy")
        if flags:
            rows.append({"idx":i,"src":src,"ref":ref,"pred":pred,
                         "flags":"|".join(flags),"len_ratio":round(ratio,3)})
    total = len(rows)
    print(f"\n  HALLUCINATION [{score_key}] [pipeline {pipeline_id}]  "
          f"{total}/{n} ({total/max(n,1)*100:.1f}%)")
    print(f"    Omission:{omission}  Repetition:{repetition}  Copy:{copy_hall}")
    if rows:
        out = os.path.join(OUTPUT_DIR, f"pipeline_{pipeline_id}",
                           f"hallucination_{score_key}.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  ✓ Hallucination log → {out}")


# ─── Loss curve ───────────────────────────────────────────────────────────────

def _plot_loss(history, fname, title):
    train_pts = [(h["epoch"],h["loss"])      for h in history if "loss"      in h]
    eval_pts  = [(h["epoch"],h["eval_loss"]) for h in history if "eval_loss" in h]
    fig, ax   = plt.subplots(figsize=(9,5))
    if train_pts: ax.plot(*zip(*train_pts), label="Train", color="#4C72B0", lw=2)
    if eval_pts:  ax.plot(*zip(*eval_pts),  label="Val",   color="#DD8452", lw=2, ls="--")
    ax.set_title(title, fontweight="bold"); ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss"); ax.legend()
    plt.tight_layout(); plt.savefig(fname); plt.close()
    print(f"  ✓ Loss curve → {fname}")


# ─── Training one pipeline ────────────────────────────────────────────────────

def _train_pipeline(pipeline_id, aug_df, dev_df, adapter_out_dir,
                    label, score_key, start_adapter_dir,
                    use_subword_reg=True):
    """Fine-tune one pipeline.  No-ops silently if a final adapter already exists."""
    if _has_final_adapter(adapter_out_dir):
        print(f"  [SKIP TRAIN] Adapter exists: {adapter_out_dir}")
        return

    resume_ckpt = _find_latest_checkpoint(adapter_out_dir)
    base_name   = get_base_model(pipeline_id)

    print(f"\n  {'='*60}")
    print(f"  TRAINING [{label}] pipeline {pipeline_id}")
    print(f"  Base        : {base_name}")
    print(f"  Start from  : {start_adapter_dir}")
    print(f"  Save to     : {adapter_out_dir}")
    print(f"  SubwordReg  : {use_subword_reg}")
    print(f"  {'='*60}")

    train_rows = aug_df.dropna(subset=["src_text", "tgt_text"]).shape[0]
    print(f"  Training rows: {train_rows:,}")
    if train_rows == 0:
        print(f"  [WARN] No training rows for pipeline {pipeline_id} — skipping.")
        return

    tokenizer    = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    model_kwargs = {"trust_remote_code": True, "torch_dtype": DTYPE}
    try:
        import flash_attn  # noqa: F401
        model_kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        pass

    base = AutoModelForSeq2SeqLM.from_pretrained(base_name, **model_kwargs)
    ip   = IndicProcessor(inference=False)

    print("  Building datasets ...")
    train_ds = _build_dataset(aug_df, pipeline_id, tokenizer, ip, base.config)
    eval_ds  = _build_dataset(dev_df, pipeline_id, tokenizer, ip, base.config)
    print(f"  Train DS: {len(train_ds):,}  |  Dev DS: {len(eval_ds):,}")

    base.gradient_checkpointing_enable()
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()

    if not os.path.isdir(start_adapter_dir):
        raise FileNotFoundError(
            f"Start adapter not found at '{start_adapter_dir}'. "
            f"Run step3_train_dora.py --pipeline {pipeline_id} first."
        )
    print(f"  Attaching start adapter from {start_adapter_dir} (is_trainable=True) ...")
    model = PeftModel.from_pretrained(base, start_adapter_dir, is_trainable=True)
    model.print_trainable_parameters()
    model = model.to(DEVICE)

    collator = _make_collator(tokenizer)

    args = Seq2SeqTrainingArguments(
        output_dir                  = adapter_out_dir,
        overwrite_output_dir        = False,
        num_train_epochs            = NUM_TRAIN_EPOCHS_AUG,
        per_device_train_batch_size = PER_DEVICE_TRAIN_BATCH,
        per_device_eval_batch_size  = PER_DEVICE_EVAL_BATCH,
        gradient_accumulation_steps = GRAD_ACCUM_STEPS,
        learning_rate               = LEARNING_RATE_AUG,
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
        args          = args,
        train_dataset = train_ds,
        eval_dataset  = eval_ds,
        data_collator = collator,
        tokenizer     = tokenizer,
        callbacks     = [EarlyStoppingCallback(
                            early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )
    trainer.train(resume_from_checkpoint=resume_ckpt)

    print(f"\n  Saving adapter → {adapter_out_dir}")
    trainer.model.save_pretrained(adapter_out_dir)

    print(f"  ✓ Saved [{label}] [pipeline {pipeline_id}]")

    fig_dir = os.path.join(FIGURES_DIR, f"pipeline_{pipeline_id}")
    os.makedirs(fig_dir, exist_ok=True)
    _plot_loss(
        trainer.state.log_history,
        os.path.join(fig_dir, f"loss_{score_key}.png"),
        f"[{label}] [pipeline {pipeline_id}]: Loss",
    )

    del model, base, trainer, train_ds, eval_ds
    gc.collect(); torch.cuda.empty_cache()


# ─── Evaluation one pipeline ──────────────────────────────────────────────────

def _eval_pipeline(pipeline_id, test_df, adapter_out_dir,
                   score_key, label, pred_rows_out, scores):
    dir_score_key = f"{score_key}_{pipeline_id}"
    if dir_score_key in scores:
        print(f"  [SKIP EVAL] {dir_score_key} already in scores.")
        return scores.get(dir_score_key)

    cfg      = PIPELINE_CONFIGS[pipeline_id]
    src_lang = cfg["src_lang"]
    tgt_lang = cfg["tgt_lang"]

    print(f"\n  {'─'*50}")
    print(f"  EVAL [{label}]  pipeline {pipeline_id}  ({cfg['label']})")
    print(f"  Adapter: {adapter_out_dir}")
    print(f"  {'─'*50}")

    sub = test_df.dropna(subset=["src_text","tgt_text"]).reset_index(drop=True)
    if len(sub) == 0:
        print(f"  [WARN] No test rows for pipeline {pipeline_id}.")
        return None

    srcs = sub["src_text"].astype(str).tolist()
    refs = sub["tgt_text"].astype(str).tolist()

    base_name    = get_base_model(pipeline_id)
    model_kwargs = {"trust_remote_code": True, "torch_dtype": DTYPE}
    try:
        import flash_attn  # noqa: F401
        model_kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        pass

    if not _has_final_adapter(adapter_out_dir):
        raise FileNotFoundError(
            f"No final adapter at '{adapter_out_dir}'. "
            f"Training must be completed before evaluation."
        )

    print("  Loading base + adapter for inference ...")
    base  = AutoModelForSeq2SeqLM.from_pretrained(base_name, **model_kwargs)
    model = PeftModel.from_pretrained(base, adapter_out_dir).to(DEVICE)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)

    ip    = IndicProcessor(inference=True)
    preds = _run_inference(srcs, model, tokenizer, ip, src_lang, tgt_lang,
                           desc=f"{score_key} pipeline {pipeline_id}")

    del model, base, tokenizer
    gc.collect(); torch.cuda.empty_cache()

    metrics = _compute_sacre(preds, refs, pipeline_id)
    comet   = _compute_comet(srcs, preds, refs)
    if comet is not None:
        metrics["COMET"] = comet
    metrics["N"] = len(preds)

    print(f"\n  TEST RESULTS [{label}] [pipeline {pipeline_id}]")
    for k, v in metrics.items():
        if k != "N":
            print(f"  {k:8s}: {v}")

    # Delta vs baseline & R1
    all_s = _load_scores()
    for ref_key in [f"baseline_{pipeline_id}", f"r1_{pipeline_id}"]:
        if ref_key in all_s:
            delta = {m: round(metrics.get(m,0)-all_s[ref_key].get(m,0), 2)
                     for m in metrics if m in all_s[ref_key] and m != "N"}
            if delta:
                tag = ref_key.replace(f"_{pipeline_id}", "")
                print(f"  Δ vs {tag:9s}: "
                      + "  ".join(f"{m}={v:+.2f}" for m,v in delta.items()))

    # Samples
    print(f"\n  Samples (first 3):")
    for i in range(min(3, len(srcs))):
        print(f"    SRC : {srcs[i][:100]}")
        print(f"    REF : {refs[i][:100]}")
        print(f"    PRED: {preds[i][:100]}\n")

    _error_analysis(preds, refs, score_key, pipeline_id)
    _hallucination_analysis(srcs, preds, refs, score_key, pipeline_id)

    for s, r, p in zip(srcs, refs, preds):
        pred_rows_out.append({"pipeline": pipeline_id,
                               "src": s, "ref": r,
                               f"{score_key}_pred": p})

    scores[dir_score_key] = metrics
    return metrics


# ─── Public entry-point ───────────────────────────────────────────────────────

def run_finetune_and_eval(
    score_key:        str,
    dataset_files:    dict,           # {pipeline_id: path | None}
    adapter_out_dirs: dict,           # {pipeline_id: path}
    pred_files:       dict,           # {pipeline_id: path}
    pipelines:        list = None,
    start_adapters:   dict = None,    # {pipeline_id: path | None}  defaults to R1
    label:            str  = "",
    use_subword_reg:  bool = True,
    eval_only_map:    dict = None,    # {pipeline_id: bool}  NEW
) -> dict:
    if pipelines is None:
        pipelines = ALL_PIPELINES
    if not label:
        label = score_key
    # Default: no pipeline is eval-only
    if eval_only_map is None:
        eval_only_map = {pid: False for pid in pipelines}

    print(f"\n{'='*65}")
    print(f"  AUG FINE-TUNE + EVAL  [{label}]")
    print(f"  Pipelines : {pipelines}")
    eval_only_pids  = [pid for pid in pipelines if eval_only_map.get(pid, False)]
    training_pids   = [pid for pid in pipelines if not eval_only_map.get(pid, False)]
    if eval_only_pids:
        print(f"  Eval-only (adapter exists) : {eval_only_pids}")
    if training_pids:
        print(f"  Train + eval               : {training_pids}")
    print(f"{'='*65}")

    if start_adapters is None:
        start_adapters = {pid: adapter_dir(pid, "r1") for pid in pipelines}

    # ── Load test split for every pipeline (always needed for eval) ────────
    test_df = {}
    for pid in pipelines:
        test_df[pid] = pd.read_excel(split_path(pid, "test"))

    # ── Load dev split only for pipelines that will train ─────────────────
    dev_df = {}
    for pid in training_pids:
        dev_df[pid] = pd.read_excel(split_path(pid, "dev"))

    # ── Fine-tune — only for pipelines that need it ────────────────────────
    for pid in training_pids:
        dataset_path = dataset_files.get(pid)
        if dataset_path is None:
            raise ValueError(
                f"dataset_files['{pid}'] is None but pipeline is in training mode. "
                f"Provide a dataset path or set eval_only_map['{pid}'] = True."
            )
        aug_df = pd.read_excel(dataset_path)
        aug_df = (aug_df.dropna(subset=["src_text","tgt_text"])
                  .sample(frac=1, random_state=SEED).reset_index(drop=True))
        _train_pipeline(
            pipeline_id       = pid,
            aug_df            = aug_df,
            dev_df            = dev_df[pid],
            adapter_out_dir   = adapter_out_dirs[pid],
            label             = label,
            score_key         = score_key,
            start_adapter_dir = start_adapters[pid],
            use_subword_reg   = use_subword_reg,
        )

    # ── Evaluate every pipeline ────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  EVALUATION  [{label}]")
    print(f"{'='*65}")

    scores    = _load_scores()
    pred_rows = {pid: [] for pid in pipelines}
    results   = {}

    for pid in pipelines:
        m = _eval_pipeline(
            pipeline_id     = pid,
            test_df         = test_df[pid],
            adapter_out_dir = adapter_out_dirs[pid],
            score_key       = score_key,
            label           = label,
            pred_rows_out   = pred_rows[pid],
            scores          = scores,
        )
        if m is not None:
            results[pid] = m
        _save_scores(scores)

    # ── Save predictions per pipeline ──────────────────────────────────────
    for pid in pipelines:
        if pred_rows[pid]:
            pd.DataFrame(pred_rows[pid]).to_csv(pred_files[pid], index=False)
            txt_path = os.path.splitext(pred_files[pid])[0] + ".txt"
            pred_col = f"{score_key}_pred"
            with open(txt_path, "w", encoding="utf-8") as f:
                for row in pred_rows[pid]:
                    f.write(row[pred_col] + "\n")
            print(f"  ✓ Predictions TXT [pipeline {pid}] → {txt_path}")
            print(f"  ✓ Predictions     [pipeline {pid}] → {pred_files[pid]}")

    _save_scores(scores)
    print(f"  ✓ Scores updated → {SCORES_FILE}")
    return results
