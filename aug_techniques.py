"""
aug_techniques.py — Augmentation Technique Generators

────────────────────────────────────────────────────────────────────────────
"""

import os, re, random, gc, unicodedata
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import nltk
from nltk.corpus import wordnet, stopwords

for _r in ["wordnet", "stopwords"]:
    nltk.download(_r, quiet=True)

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

try:
    from IndicTransToolkit.processor import IndicProcessor
except ImportError:
    raise ImportError(
        "IndicTransToolkit required.\n"
        "  pip install git+https://github.com/VarunGumma/IndicTransToolkit.git"
    )

from config import (
    ALL_PIPELINES, PIPELINE_CONFIGS,
    SEED, BF16, FP16,
    INFER_BATCH_SIZE, NUM_BEAMS, MAX_NEW_TOKENS,
    MAX_SEQ_LEN, LENGTH_PENALTY, NO_REPEAT_NGRAM_SIZE,
    BT_LABSE_THRESHOLD, BT_SAMPLE_RATIO,
    FT_AUG_SAMPLE_RATIO, AUG_SR_PROB, AUG_N_PER_SENT, QA_LABSE_THRESHOLD,
    ITER_SAMPLE_RATIO, ST_CONF_THRESHOLD,
    NOISE_SAMPLE_RATIO, NOISE_SWAP_PROB, NOISE_DELETE_PROB, NOISE_INSERT_PROB,
    get_base_model, adapter_dir, synth_cache_path,
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if BF16 else (torch.float16 if FP16 else torch.float32)

STOP_WORDS = set(stopwords.words("english"))

_LANG_TAG_RE = re.compile(r'^(?:[a-z]{3}_[A-Za-z]{4}\s+){1,2}')

# ── Which pipeline provides the BT model for each pipeline ────────────────────
# BT for pipeline X uses the reverse-direction pipeline's R1 adapter.
_BT_PARTNER = {
    "A": "B",  # eng→mni_Mtei  BT needs mni_Mtei→eng  (pipeline B)
    "B": "A",  # mni_Mtei→eng  BT needs eng→mni_Mtei  (pipeline A)
    "C": "D",  # eng→mni_Beng  BT needs mni_Beng→eng  (pipeline D)
    "D": "C",  # mni_Beng→eng  BT needs eng→mni_Beng  (pipeline C)
}


# ─── Language-tag stripping ───────────────────────────────────────────────────

def _strip_lang_tag(text: str) -> str:
    return _LANG_TAG_RE.sub("", text).strip()

def _strip_lang_tags(texts: list) -> list:

    return [unicodedata.normalize("NFC", _strip_lang_tag(t)) for t in texts]


# ─── IT2 model loading ────────────────────────────────────────────────────────

def _load_it2_model(pipeline_id: str, adapter_path: str | None = None):

    base_name    = get_base_model(pipeline_id)
    model_kwargs = {"trust_remote_code": True, "torch_dtype": DTYPE}
    try:
        import flash_attn  # noqa: F401
        model_kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        pass

    print(f"  [IT2] Loading base: {base_name}  (pipeline {pipeline_id})")
    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    base      = AutoModelForSeq2SeqLM.from_pretrained(base_name, **model_kwargs)

    if adapter_path is not None:
        if not os.path.isdir(adapter_path):
            raise FileNotFoundError(
                f"Adapter not found at {adapter_path}. "
                f"Run step3_train_lora.py --pipeline {pipeline_id} first."
            )
        print(f"  [IT2] Attaching adapter: {adapter_path}")
        model = PeftModel.from_pretrained(base, adapter_path)
    else:
        model = base

    model = model.to(DEVICE)
    model.eval()
    ip = IndicProcessor(inference=True)
    return model, tokenizer, ip


def _free_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─── Core translate helpers ───────────────────────────────────────────────────

def _translate_batch(sentences, model, tokenizer, ip, src_lang, tgt_lang):
    batch  = ip.preprocess_batch(sentences, src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tokenizer(
        batch, truncation=True, padding="longest",
        return_tensors="pt", max_length=MAX_SEQ_LEN,
    ).to(DEVICE)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            use_cache=True, min_length=0,
            max_new_tokens=MAX_NEW_TOKENS, num_beams=NUM_BEAMS,
            length_penalty=LENGTH_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            early_stopping=True,
        )

    with tokenizer.as_target_tokenizer():
        decoded = tokenizer.batch_decode(
            generated.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    translations = ip.postprocess_batch(decoded, lang=tgt_lang)
    return _strip_lang_tags(translations)


def _batch_translate(sentences, model, tokenizer, ip, src_lang, tgt_lang,
                     batch_size=INFER_BATCH_SIZE, desc="Translating"):
    all_preds = []
    for i in tqdm(range(0, len(sentences), batch_size), desc=f"  [{desc}]"):
        chunk = sentences[i:i+batch_size]
        preds = _translate_batch(chunk, model, tokenizer, ip, src_lang, tgt_lang)
        all_preds.extend(preds)
    return all_preds


# ─── LaBSE quality filter ─────────────────────────────────────────────────────

def _labse_filter(src_texts: list, tgt_texts: list, threshold: float):
    try:
        from sentence_transformers import SentenceTransformer
        print(f"  [LaBSE] Scoring {len(src_texts):,} pairs (threshold={threshold}) ...")
        labse   = SentenceTransformer("sentence-transformers/LaBSE")
        src_emb = labse.encode(src_texts, batch_size=128,
                               normalize_embeddings=True, show_progress_bar=False)
        tgt_emb = labse.encode(tgt_texts, batch_size=128,
                               normalize_embeddings=True, show_progress_bar=False)
        scores  = (src_emb * tgt_emb).sum(axis=1)
        mask    = scores >= threshold
        print(f"  [LaBSE] Kept {mask.sum():,}/{len(src_texts):,} "
              f"(mean={scores.mean():.3f})")
        del labse; gc.collect()
        return mask, scores
    except ImportError:
        print("  [LaBSE] sentence-transformers not installed — skipping filter.")
        scores = np.ones(len(src_texts), dtype=np.float32)
        return scores >= 0.0, scores


# ─── Cache helpers ────────────────────────────────────────────────────────────

def _load_or_generate(pipeline_id: str, technique: str,
                      generator_fn, *args, **kwargs) -> pd.DataFrame:
    cache = synth_cache_path(pipeline_id, technique)
    if os.path.exists(cache):
        df = pd.read_excel(cache)
        print(f"  [CACHE HIT] pipeline_{pipeline_id}/{technique}: "
              f"{len(df):,} pairs from {cache}")
        return df
    print(f"  [CACHE MISS] pipeline_{pipeline_id}/{technique}: generating ...")
    df = generator_fn(*args, **kwargs)
    if len(df) > 0:
        df.to_excel(cache, index=False)
        print(f"  [CACHED] {len(df):,} pairs → {cache}")
    return df


# ─── Synonym utilities (English source side) ─────────────────────────────────

def _get_synonyms(word: str) -> list:
    syns = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            c = lemma.name().replace("_", " ")
            if c.lower() != word.lower() and " " not in c:
                syns.add(c)
    return list(syns)


def _synonym_replace(sentence: str) -> str:
    words     = sentence.split()
    n_replace = max(1, int(len(words) * min(AUG_SR_PROB, 0.15)))
    cands     = [i for i, w in enumerate(words)
                 if len(w) > 1 and w.lower() not in STOP_WORDS and _get_synonyms(w)]
    random.shuffle(cands)
    for idx in cands[:n_replace]:
        syns = _get_synonyms(words[idx])
        if syns:
            words[idx] = random.choice(syns)
    return " ".join(words)


# ─── Character-level perturbation (Indic script sources) ─────────────────────

def _char_perturb(text: str, swap_prob: float = 0.10) -> str:
    """Swap adjacent characters with probability swap_prob."""
    chars = list(text)
    n     = len(chars)
    if n < 4:
        return text
    for i in range(n - 1):
        if random.random() < swap_prob:
            chars[i], chars[i+1] = chars[i+1], chars[i]
    return "".join(chars)


# ─── Noise injection helpers (Step 5d) ───────────────────────────────────────

def _noise_inject(text: str, vocab: list | None = None) -> str:
    """Apply word-level noise: swap, delete, or insert random words."""
    words = text.split()
    if len(words) < 2:
        return text

    # Word swap
    new_words = words[:]
    for i in range(len(new_words) - 1):
        if random.random() < NOISE_SWAP_PROB:
            new_words[i], new_words[i+1] = new_words[i+1], new_words[i]

    # Word deletion
    new_words = [w for w in new_words if random.random() > NOISE_DELETE_PROB]
    if not new_words:
        new_words = words[:]   # don't delete everything

    # Word insertion (random word from local vocabulary)
    if vocab:
        result = []
        for w in new_words:
            result.append(w)
            if random.random() < NOISE_INSERT_PROB:
                result.append(random.choice(vocab))
        new_words = result

    return " ".join(new_words)


def _build_vocab(texts: list) -> list:
    vocab = list({w for t in texts for w in str(t).split() if len(w) > 2})
    return vocab


# ─── TECHNIQUE 1: Back-Translation ───────────────────────────────────────────

def _run_bt(train_df: pd.DataFrame, pipeline_id: str) -> pd.DataFrame:

    cfg         = PIPELINE_CONFIGS[pipeline_id]
    partner_id  = _BT_PARTNER[pipeline_id]
    partner_cfg = PIPELINE_CONFIGS[partner_id]

    df = train_df.dropna(subset=["src_text", "tgt_text"]).reset_index(drop=True)
    n  = min(int(len(df) * BT_SAMPLE_RATIO), len(df))
    sample = df.sample(n=n, random_state=SEED).reset_index(drop=True)

    # BT: translate the TARGET of this pipeline → source language
    # using the PARTNER pipeline's model (which goes target_lang→source_lang)
    bt_src_lang = partner_cfg["src_lang"]   # = this pipeline's tgt_lang
    bt_tgt_lang = partner_cfg["tgt_lang"]   # = this pipeline's src_lang
    bt_inputs   = sample["tgt_text"].astype(str).tolist()

    print(f"\n  [BT] pipeline {pipeline_id}: back-translating {n:,} "
          f"'{cfg['tgt_lang']}' → '{cfg['src_lang']}' "
          f"using pipeline {partner_id} R1 adapter")

    partner_adapter = adapter_dir(partner_id, "r1")
    model, tokenizer, ip = _load_it2_model(partner_id, adapter_path=partner_adapter)

    bt_outputs = _batch_translate(
        bt_inputs, model, tokenizer, ip, bt_src_lang, bt_tgt_lang,
        desc=f"BT pipeline {pipeline_id} via {partner_id}",
    )
    _free_model(model)

    # Validity filter
    valid = [(bt_outputs[i], sample["tgt_text"].iloc[i])
             for i in range(len(bt_outputs))
             if bt_outputs[i].strip() and len(bt_outputs[i].split()) >= 3]
    if not valid:
        print("  [BT] No valid pairs — returning empty DataFrame.")
        return pd.DataFrame(columns=["src_text","tgt_text","aug_type","qa_score"])

    v_src_bt, v_tgt_orig = zip(*valid)
    # LaBSE: compare bt_src vs original src (same script → meaningful cosine)
    orig_src = [sample["src_text"].iloc[i]
                for i in range(len(bt_outputs))
                if bt_outputs[i].strip() and len(bt_outputs[i].split()) >= 3]

    mask, scores = _labse_filter(list(v_src_bt), orig_src, BT_LABSE_THRESHOLD)

    new_src = [v_src_bt[i]   for i in range(len(v_src_bt))  if mask[i]]
    new_tgt = [v_tgt_orig[i] for i in range(len(v_tgt_orig)) if mask[i]]

    df_out = pd.DataFrame({
        "src_text": new_src,
        "tgt_text": new_tgt,
        "aug_type": "back_translation",
        "qa_score": scores[mask],
        "pipeline": pipeline_id,
    })
    df_out = df_out[
        df_out["src_text"].str.strip().ne("") &
        df_out["tgt_text"].str.strip().ne("")
    ].reset_index(drop=True)

    print(f"  [BT] Final pairs: {len(df_out):,}")
    return df_out


def generate_bt(train_df: pd.DataFrame, pipeline_id: str) -> pd.DataFrame:

    return _load_or_generate(pipeline_id, "bt", _run_bt, train_df, pipeline_id)


# ─── TECHNIQUE 2: Forward Translation ────────────────────────────────────────

def _run_ft_aug(train_df: pd.DataFrame, pipeline_id: str) -> pd.DataFrame:

    cfg      = PIPELINE_CONFIGS[pipeline_id]
    src_lang = cfg["src_lang"]
    tgt_lang = cfg["tgt_lang"]

    df = train_df.dropna(subset=["src_text", "tgt_text"]).reset_index(drop=True)
    n  = min(int(len(df) * FT_AUG_SAMPLE_RATIO), len(df))
    sample = df.sample(n=n, random_state=SEED+20).reset_index(drop=True)

    eng_source = src_lang == "eng_Latn"

    if eng_source:
        # Synonym-paraphrase English sources
        print(f"  [FT-Aug] Synonym-paraphrasing {n:,} English sources "
              f"(pipeline {pipeline_id}) ...")
        para_src = []
        for src in tqdm(sample["src_text"].astype(str),
                        desc="  Synonym paraphrase", leave=False):
            seen     = {src.strip()}
            variants = []
            for _ in range(AUG_N_PER_SENT * 6):
                aug = _synonym_replace(src)
                if aug.strip() and aug.strip() not in seen:
                    variants.append(aug); seen.add(aug.strip())
                if variants:
                    break
            para_src.append(variants[0] if variants else src)
    else:
        # Character-perturb Indic sources
        print(f"  [FT-Aug] Character-perturbing {n:,} Indic sources "
              f"(pipeline {pipeline_id}) ...")
        para_src = [_char_perturb(str(s)) for s in
                    tqdm(sample["src_text"], desc="  Char perturb", leave=False)]

    print(f"  [FT-Aug] Translating {len(para_src):,} paraphrased sources → "
          f"{tgt_lang} using pipeline {pipeline_id} R1 adapter ...")
    r1_adapter = adapter_dir(pipeline_id, "r1")
    model, tokenizer, ip = _load_it2_model(pipeline_id, adapter_path=r1_adapter)

    tgt_preds = _batch_translate(
        para_src, model, tokenizer, ip, src_lang, tgt_lang,
        desc=f"FT-Aug pipeline {pipeline_id}",
    )
    _free_model(model)

    # Validity filter
    valid = [(para_src[i], tgt_preds[i])
             for i in range(len(tgt_preds))
             if para_src[i].strip() and tgt_preds[i].strip()
             and len(tgt_preds[i].split()) >= 2]
    if not valid:
        print("  [FT-Aug] No valid pairs after filtering.")
        return pd.DataFrame(columns=["src_text","tgt_text","aug_type","qa_score"])

    v_src, v_tgt = zip(*valid)
    mask, scores = _labse_filter(list(v_src), list(v_tgt), QA_LABSE_THRESHOLD)

    df_out = pd.DataFrame({
        "src_text": [v_src[i] for i in range(len(v_src)) if mask[i]],
        "tgt_text": [v_tgt[i] for i in range(len(v_tgt)) if mask[i]],
        "aug_type": "forward_translation",
        "qa_score": scores[mask],
        "pipeline": pipeline_id,
    })
    df_out = df_out[
        df_out["src_text"].str.strip().ne("") &
        df_out["tgt_text"].str.strip().ne("")
    ].reset_index(drop=True)

    print(f"  [FT-Aug] Final pairs: {len(df_out):,}")
    return df_out


def generate_ft_aug(train_df: pd.DataFrame, pipeline_id: str) -> pd.DataFrame:
    """Public entry-point for Forward Translation augmentation."""
    return _load_or_generate(pipeline_id, "ft_aug", _run_ft_aug, train_df, pipeline_id)


# ─── TECHNIQUE 3: Iterative Pseudo-labelling ─────────────────────────────────

def _run_iter(train_df: pd.DataFrame, pipeline_id: str,
              prev_adapter_dir: str) -> pd.DataFrame:

    cfg      = PIPELINE_CONFIGS[pipeline_id]
    src_lang = cfg["src_lang"]
    tgt_lang = cfg["tgt_lang"]

    df = train_df.dropna(subset=["src_text", "tgt_text"]).reset_index(drop=True)
    n  = min(int(len(df) * ITER_SAMPLE_RATIO), len(df))
    sample = df.sample(n=n, random_state=SEED+30).reset_index(drop=True)

    print(f"\n  [Iter] Pseudo-labelling {n:,} sources for pipeline {pipeline_id} "
          f"using {prev_adapter_dir} ...")

    model, tokenizer, ip = _load_it2_model(pipeline_id,
                                            adapter_path=prev_adapter_dir)
    inputs    = sample["src_text"].astype(str).tolist()
    gold_refs = sample["tgt_text"].astype(str).tolist()

    pseudo = _batch_translate(
        inputs, model, tokenizer, ip, src_lang, tgt_lang,
        desc=f"Iter pseudo pipeline {pipeline_id}",
    )
    _free_model(model)

    # Validity: non-empty, ≥3 tokens, not identical to gold
    valid_idx = [i for i in range(len(pseudo))
                 if (pseudo[i].strip()
                     and len(pseudo[i].split()) >= 3
                     and gold_refs[i].strip()
                     and pseudo[i].strip() != gold_refs[i].strip())]
    if not valid_idx:
        print("  [Iter] No valid pseudo-label pairs.")
        return pd.DataFrame(columns=["src_text","tgt_text","aug_type","qa_score"])

    v_pseudo = [pseudo[i]    for i in valid_idx]
    v_gold   = [gold_refs[i] for i in valid_idx]
    v_input  = [inputs[i]    for i in valid_idx]

    mask, scores = _labse_filter(v_pseudo, v_gold, ST_CONF_THRESHOLD)

    df_out = pd.DataFrame({
        "src_text": [v_input[i]  for i in range(len(v_input))  if mask[i]],
        "tgt_text": [v_pseudo[i] for i in range(len(v_pseudo)) if mask[i]],
        "aug_type": "iter_pseudo",
        "qa_score": scores[mask],
        "pipeline": pipeline_id,
    })
    df_out = df_out[
        df_out["src_text"].str.strip().ne("") &
        df_out["tgt_text"].str.strip().ne("")
    ].reset_index(drop=True)

    print(f"  [Iter] Final pseudo pairs: {len(df_out):,}")
    return df_out


def generate_iter(train_df: pd.DataFrame, pipeline_id: str,
                  prev_adapter_dir: str) -> pd.DataFrame:
    """Public entry-point for Iterative Pseudo-labelling."""
    return _load_or_generate(pipeline_id, "iter", _run_iter,
                             train_df, pipeline_id, prev_adapter_dir)


# ─── TECHNIQUE 4: Noise Injection (source side only) ─────────────────────────

def _run_noise(train_df: pd.DataFrame, pipeline_id: str) -> pd.DataFrame:
    """
    Noise Injection on the source side only (Step 5d).

    Applies word-level noise (swap / delete / insert) to source text.
    Target text is kept as-is (gold).  No model inference needed.
    Produces: (noisy_source, original_target)

    The noisy pairs teach the model to be robust to input noise.
    """
    cfg = PIPELINE_CONFIGS[pipeline_id]

    df = train_df.dropna(subset=["src_text", "tgt_text"]).reset_index(drop=True)
    n  = min(int(len(df) * NOISE_SAMPLE_RATIO), len(df))
    sample = df.sample(n=n, random_state=SEED+40).reset_index(drop=True)

    sources = sample["src_text"].astype(str).tolist()
    targets = sample["tgt_text"].astype(str).tolist()

    print(f"\n  [Noise] Injecting noise into {n:,} sources "
          f"for pipeline {pipeline_id} ...")

    vocab       = _build_vocab(sources)
    noisy_srcs  = [_noise_inject(s, vocab) for s in
                   tqdm(sources, desc="  Noise injection", leave=False)]

    # Filter: remove rows where noise made src identical to original or empty
    rows = [(ns, t, s) for ns, t, s in zip(noisy_srcs, targets, sources)
            if ns.strip() and ns.strip() != s.strip()]

    if not rows:
        print("  [Noise] No valid noisy pairs.")
        return pd.DataFrame(columns=["src_text","tgt_text","aug_type","qa_score"])

    v_noisy, v_tgt, _ = zip(*rows)

    # Optionally filter with LaBSE (noisy src vs original src, same language)
    orig_src = [s for _, _, s in rows]
    mask, scores = _labse_filter(list(v_noisy), orig_src, BT_LABSE_THRESHOLD)

    df_out = pd.DataFrame({
        "src_text": [v_noisy[i] for i in range(len(v_noisy)) if mask[i]],
        "tgt_text": [v_tgt[i]   for i in range(len(v_tgt))   if mask[i]],
        "aug_type": "noise_injection",
        "qa_score": scores[mask],
        "pipeline": pipeline_id,
    })
    df_out = df_out[
        df_out["src_text"].str.strip().ne("") &
        df_out["tgt_text"].str.strip().ne("")
    ].reset_index(drop=True)

    print(f"  [Noise] Final noisy pairs: {len(df_out):,}")
    return df_out


def generate_noise(train_df: pd.DataFrame, pipeline_id: str) -> pd.DataFrame:
    """Public entry-point for Noise Injection augmentation."""
    return _load_or_generate(pipeline_id, "noise", _run_noise,
                             train_df, pipeline_id)
