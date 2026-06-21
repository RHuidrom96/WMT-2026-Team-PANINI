"""
translate_inference.py — Translate new (unlabeled) files using trained adapters

"""

import os, re, gc, argparse, warnings, unicodedata
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

warnings.filterwarnings("ignore")

from config import (
    ALL_PIPELINES, PIPELINE_CONFIGS,
    INFER_BATCH_SIZE, NUM_BEAMS, MAX_NEW_TOKENS, MAX_SEQ_LEN,
    LENGTH_PENALTY, NO_REPEAT_NGRAM_SIZE,
    BF16, FP16, SEED,
    adapter_dir, get_base_model,
)

try:
    from config import REPETITION_PENALTY
except ImportError:
    REPETITION_PENALTY = 1.3

torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if BF16 else (torch.float16 if FP16 else torch.float32)

try:
    from IndicTransToolkit.processor import IndicProcessor
except ImportError:
    raise ImportError(
        "IndicTransToolkit required. Install:\n"
        "  pip install git+https://github.com/VarunGumma/IndicTransToolkit.git"
    )

_LANG_TAG_RE = re.compile(r'^(?:[a-z]{3}_[A-Za-z]{4}\s+){1,2}', re.IGNORECASE)

_ASCII_TO_MTEI = str.maketrans("0123456789", "꯰꯱꯲꯳꯴꯵꯶꯷꯸꯹")
_ASCII_TO_BENG = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")

def _fix_digits(text: str, tgt_lang: str) -> str:
    if tgt_lang == "mni_Mtei":
        return text.translate(_ASCII_TO_MTEI)
    elif tgt_lang == "mni_Beng":
        return text.translate(_ASCII_TO_BENG)
    return text


# ─── FIX 5: Residual English letters leaking into eng→mni_Mtei / eng→mni_Beng ─
_ENG_LETTER_MTEI = {
    'A': 'ꯑꯦ',  'B': 'ꯕꯤ',   'C': 'ꯁꯤ',      'D': 'ꯗꯤ',   'E': 'ꯏ',
    'F': 'ꯐ',   'G': 'ꯖꯤ',   'H': 'ꯦꯏꯆ',    'I': 'ꯑꯥꯏ', 'J': 'ꯖꯦ',
    'K': 'ꯀꯦ',  'L': 'ꯑꯦꯜ', 'M': 'ꯑꯦꯝ',    'N': 'ꯑꯦꯟ', 'O': 'ꯑꯣ',
    'P': 'ꯄꯤ',  'Q': 'ꯀ꯭ꯌꯨ','R': 'ꯑꯥꯔ',    'S': 'ꯑꯦꯁ', 'T': 'ꯇꯤ',
    'U': 'ꯌꯨ',  'V': 'ꯚꯤ',   'W': 'ꯗꯕ꯭ꯂꯌꯨ','X': 'ꯑꯦꯛꯁ','Y': 'ꯋꯥꯏ',
    'Z': 'ꯖꯦꯗ',
}

_ENG_LETTER_BENG = {
    'A': 'এ',    'B': 'বি',   'C': 'সি',   'D': 'ডি',   'E': 'ই',
    'F': 'এফ',  'G': 'জি',   'H': 'এইচ', 'I': 'আই',  'J': 'জে',
    'K': 'কে',  'L': 'এল',  'M': 'এম',   'N': 'এন',   'O': 'ও',
    'P': 'পি',  'Q': 'কিউ', 'R': 'আর',   'S': 'এস',   'T': 'টি',
    'U': 'ইউ',  'V': 'ভি',   'W': 'ডাবলিউ','X': 'এক্স','Y': 'ওয়াই',
    'Z': 'জেড',
}


def _transliterate_english_chars(text: str, tgt_lang: str) -> str:
    """
    Convert any residual ASCII English letters in a translated Meitei string
    into their spoken-name equivalents in the target script.
    Only acts on mni_Mtei / mni_Beng targets (pipelines A and C).
    For eng_Latn targets (pipelines B and D) this is a no-op.
    """
    if tgt_lang == "mni_Mtei":
        letter_map = _ENG_LETTER_MTEI
    elif tgt_lang == "mni_Beng":
        letter_map = _ENG_LETTER_BENG
    else:
        return text                        # eng_Latn target — left untouched

    if not re.search(r'[A-Za-z]', text):   # fast-path: nothing to do
        return text

    def _replace(m: re.Match) -> str:
        return ''.join(letter_map.get(ch.upper(), ch) for ch in m.group(0))

    return re.sub(r'[A-Za-z]+', _replace, text)


# ─── FIX 1: Source preprocessing — strip reference/footnote tags ──────────────
_REF_TAG_RE = re.compile(
    r'<[^>]{1,30}?\s+\d+\s*>'   # <word 1>, <ꯑꯥꯏ ꯗꯤ ꯱>, etc.
    r'|\[\d+\]'                  # [1], [23]
    r'|\(\d+\)',                 # (1), (23)
    re.UNICODE,
)

def _strip_ref_tags(text: str) -> str:
    """Remove reference/footnote markers from source text before translation."""
    return re.sub(r'\s+', ' ', _REF_TAG_RE.sub('', text)).strip()


# ─── FIX 6: Normalize en-dash/em-dash in numeric ranges in source text ────────

_NUMERIC_DASH_RE = re.compile(r'(\d)\s*[–—]\s*(\d)')  # en-dash U+2013, em-dash U+2014

def _normalize_numeric_dashes(text: str) -> str:
    """Replace en-dash/em-dash between digits with a plain ASCII hyphen."""
    return _NUMERIC_DASH_RE.sub(r'\1-\2', text)


# ─── FIX 2: Post-processing — strip leaked script fragments from English output ─
_NON_LATIN_SCRIPT_RE = re.compile(
    r'(?:[\u0900-\u097F\u0980-\u09FF\uAAE0-\uAAFF\uABC0-\uABFF]'
    r'[\u0900-\u097F\u0980-\u09FF\uAAE0-\uAAFF\uABC0-\uABFF\u09CD\u0ACD\u0BCD.'
    r'\u200C\u200D]*)+',
    re.UNICODE,
)

def _strip_source_script_from_english(text: str) -> str:
    """
    Remove Brahmic/Meitei script fragments leaked into an English translation.
    Returns the cleaned text (caller is responsible for flagging the row).
    """
    cleaned = _NON_LATIN_SCRIPT_RE.sub('', text)
    return re.sub(r'\s+', ' ', cleaned).strip()


# ─── Script-mixing QA check ────────────────────────────────────────────────

_DEVANAGARI_LETTER_RE = re.compile(
    r'[\u0904-\u0963\u0966-\u096F\u0971-\u097F]',
    re.UNICODE,
)
_BENGALI_RE = re.compile(r'[\u0980-\u09FF]')
_MTEI_RE    = re.compile(r'[\uAAE0-\uAAFF\uABC0-\uABFF]')

# Excludes U+0964/U+0965 (danda/double-danda) so shared punctuation is preserved.
_DEVANAGARI_RUN_RE = re.compile(
    r'[\u0904-\u0963\u0966-\u096F\u0971-\u097F]'
    r'[\u0900-\u0963\u0966-\u097F\u0902\u0903\u093C\u093E-\u094C\u094D\u0951-\u0954]*',
    re.UNICODE,
)

_DEVANAGARI_RE = _DEVANAGARI_LETTER_RE


def _strip_devanagari_from_meitei(text: str) -> str:

    cleaned = _DEVANAGARI_RUN_RE.sub('', text)
    return re.sub(r'\s+', ' ', cleaned).strip()


def _script_issue(text: str, tgt_lang: str) -> str:

    if not text:
        return ""
    if tgt_lang in ("mni_Mtei", "mni_Beng"):
        # Use letter-only check — danda/punctuation shared codepoints are NOT leaks
        if _DEVANAGARI_LETTER_RE.search(text):
            return "devanagari_leak"
    elif tgt_lang == "eng_Latn":
        if _MTEI_RE.search(text) or _BENGALI_RE.search(text) or _DEVANAGARI_LETTER_RE.search(text):
            return "source_script_leak"
    return ""


def _strip_lang_tag(text: str) -> str:
    return _LANG_TAG_RE.sub("", text).strip()


# ─── FIX 7: Capitalize the first letter of English translations ───────────────

def _capitalize_first(text: str) -> str:

    for i, ch in enumerate(text):
        if ch.isalpha():
            if ch.islower():
                return text[:i] + ch.upper() + text[i + 1:]
            return text   # first alpha is already uppercase — no-op
    return text           # no alphabetic character found


_CONSEC_DUP_RE = re.compile(r'\b(\S+)( \1)+\b', re.IGNORECASE)

def _remove_consecutive_duplicates(text: str) -> str:
    """
    Collapse consecutive duplicate word runs: "stomach stomach" → "stomach",
    "the the the" → "the". Case-insensitive match, case of FIRST token kept.
    """
    return _CONSEC_DUP_RE.sub(r'\1', text)


# ─── Degenerate-repetition QA check ────────────────────────────────────────
_REPEAT_RUN_THRESHOLD = 4
_REPEAT_NGRAM_LENS     = (1, 2, 3)


def _find_degenerate_repetition(text: str):

    words = text.split(" ")
    n = len(words)
    for gram_len in _REPEAT_NGRAM_LENS:
        i = 0
        while i + gram_len * _REPEAT_RUN_THRESHOLD <= n:
            gram = words[i:i + gram_len]
            reps = 1
            j = i + gram_len
            while j + gram_len <= n and words[j:j + gram_len] == gram:
                reps += 1
                j += gram_len
            if reps >= _REPEAT_RUN_THRESHOLD:
                return True, " ".join(words[:i]).strip()
            i += 1
    return False, text


def _strip_lang_tags(texts: list) -> list:
    return [_strip_lang_tag(t) for t in texts]


def _clean_text(text) -> str:

    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r" *꯭ *", "꯭", text)
    return text


# ─── FIX 3: Incomplete translation detection ──────────────────────────────────
_SHORT_OUTPUT_RATIO = 0.25
_SHORT_OUTPUT_MIN_SRC_WORDS = 5

def _is_suspiciously_short(src: str, hyp: str) -> bool:
    src_words = len(src.split())
    hyp_words = len(hyp.split())
    if src_words < _SHORT_OUTPUT_MIN_SRC_WORDS:
        return False
    if hyp_words == 0:
        return True
    return (hyp_words / src_words) < _SHORT_OUTPUT_RATIO


# ─── Inference ────────────────────────────────────────────────────────────────

def _generate(sentences, model, tokenizer, ip, src_lang, tgt_lang,
               num_beams, repetition_penalty, no_repeat_ngram_size,
               early_stopping=True, max_new_tokens=MAX_NEW_TOKENS):
    """One generate() + decode + postprocess pass. Returns cleaned translations."""
    batch  = ip.preprocess_batch(sentences, src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tokenizer(
        batch, truncation=True, padding="longest",
        return_tensors="pt", max_length=MAX_SEQ_LEN,
    ).to(DEVICE)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            use_cache=True, min_length=0,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            length_penalty=LENGTH_PENALTY,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            early_stopping=early_stopping,
        )

    with tokenizer.as_target_tokenizer():
        decoded = tokenizer.batch_decode(
            generated.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    translations = ip.postprocess_batch(decoded, lang=tgt_lang)
    translations = _strip_lang_tags(translations)
    translations = [_fix_digits(t, tgt_lang) for t in translations]

    translations = [_transliterate_english_chars(t, tgt_lang) for t in translations]

    if tgt_lang == "eng_Latn":
        translations = [_capitalize_first(t) for t in translations]

    if tgt_lang == "eng_Latn":
        translations = [_remove_consecutive_duplicates(t) for t in translations]
    return [_clean_text(t) for t in translations]


def translate_batch(sentences, model, tokenizer, ip, src_lang, tgt_lang,
                     num_beams=NUM_BEAMS, repetition_penalty=REPETITION_PENALTY):
    translations = _generate(sentences, model, tokenizer, ip, src_lang, tgt_lang,
                              num_beams, repetition_penalty, NO_REPEAT_NGRAM_SIZE)

    # ── Degenerate repetition check & retry ──────────────────────────────────
    checks = [_find_degenerate_repetition(t) for t in translations]
    retry_idx = [i for i, (is_degen, _) in enumerate(checks) if is_degen]

    if retry_idx:
        retry_sents = [sentences[i] for i in retry_idx]
        retry_out = _generate(
            retry_sents, model, tokenizer, ip, src_lang, tgt_lang,
            num_beams=1,
            repetition_penalty=max(repetition_penalty + 0.5, 1.8),
            no_repeat_ngram_size=2,
        )
        for pos, i in enumerate(retry_idx):
            retry_is_degen, _ = _find_degenerate_repetition(retry_out[pos])
            if not retry_is_degen:
                translations[i] = retry_out[pos]
                checks[i] = (False, retry_out[pos])

    # ── FIX 3: Short/incomplete output check & retry ─────────────────────────
    short_idx = [
        i for i, (t, s) in enumerate(zip(translations, sentences))
        if not checks[i][0]
        and _is_suspiciously_short(s, t)
    ]
    if short_idx:
        short_sents = [sentences[i] for i in short_idx]
        short_retry = _generate(
            short_sents, model, tokenizer, ip, src_lang, tgt_lang,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            early_stopping=False,
            max_new_tokens=int(MAX_NEW_TOKENS * 1.5),
        )
        for pos, i in enumerate(short_idx):
            retry_is_degen, _ = _find_degenerate_repetition(short_retry[pos])
            if not retry_is_degen and not _is_suspiciously_short(sentences[i], short_retry[pos]):
                translations[i] = short_retry[pos]

    # ── FIX 2: Strip leaked script from English output ───────────────────────
    if tgt_lang == "eng_Latn":
        for i, t in enumerate(translations):
            if _script_issue(t, tgt_lang):
                translations[i] = _strip_source_script_from_english(t)

    # ── FIX 4: Devanagari leak in Meitei output — retry then strip ───────────
    if tgt_lang == "mni_Mtei":
        deva_idx = [
            i for i, t in enumerate(translations)
            if not checks[i][0]
            and _DEVANAGARI_RE.search(t)
        ]
        if deva_idx:
            deva_sents = [sentences[i] for i in deva_idx]
            deva_retry = _generate(
                deva_sents, model, tokenizer, ip, src_lang, tgt_lang,
                num_beams=1,
                repetition_penalty=max(repetition_penalty + 0.3, 1.6),
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                early_stopping=False,
            )
            for pos, i in enumerate(deva_idx):
                retry_is_degen, _ = _find_degenerate_repetition(deva_retry[pos])
                if retry_is_degen:
                    translations[i] = _strip_devanagari_from_meitei(translations[i])
                    checks[i] = (False, translations[i])
                elif not _DEVANAGARI_RE.search(deva_retry[pos]):
                    translations[i] = deva_retry[pos]
                    checks[i] = (False, deva_retry[pos])
                else:
                    translations[i] = _strip_devanagari_from_meitei(deva_retry[pos])
                    checks[i] = (False, translations[i])

    # ── Build flag column ─────────────────────────────────────────────────────
    flags = []
    for i, t in enumerate(translations):
        is_degen, clean_t = checks[i]
        if is_degen:
            translations[i] = clean_t
        issue = _script_issue(translations[i], tgt_lang)
        tag = "degenerate_repetition" if is_degen else ""
        short_tag = "short_output" if _is_suspiciously_short(sentences[i], translations[i]) else ""
        flags.append("|".join(x for x in (tag, issue, short_tag) if x))

    return translations, flags


def translate_sentences(sentences, model, tokenizer, ip, src_lang, tgt_lang,
                         batch_size=INFER_BATCH_SIZE, num_beams=NUM_BEAMS,
                         repetition_penalty=REPETITION_PENALTY):

    n = len(sentences)
    outputs = [""] * n
    flags   = [""] * n

    nonempty_idx   = [i for i, s in enumerate(sentences) if s]
    nonempty_sents = [sentences[i] for i in nonempty_idx]

    if not nonempty_sents:
        return outputs, flags

    for start in tqdm(range(0, len(nonempty_sents), batch_size),
                       desc=f"  Translating ({src_lang}→{tgt_lang})"):
        chunk = nonempty_sents[start:start + batch_size]
        preds, pred_flags = translate_batch(chunk, model, tokenizer, ip,
                                             src_lang, tgt_lang, num_beams=num_beams,
                                             repetition_penalty=repetition_penalty)
        for j, (p, f) in enumerate(zip(preds, pred_flags)):
            outputs[nonempty_idx[start + j]] = p
            flags[nonempty_idx[start + j]]   = f

    return outputs, flags


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _find_text_column(df: pd.DataFrame, text_col: str = None) -> str:
    if text_col:
        if text_col in df.columns:
            return text_col
        raise ValueError(
            f"--text-col '{text_col}' not found in file. "
            f"Available columns: {list(df.columns)}"
        )
    for cand in ["src_text", "source", "text", "sentence", "input"]:
        if cand in df.columns:
            return cand
    first = df.columns[0]
    print(f"  [WARN] No standard text column found — using first column '{first}'. "
          f"Pass --text-col to override.")
    return first


def _adapter_ready(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    has_cfg = os.path.exists(os.path.join(path, "adapter_config.json"))
    has_wts = (os.path.exists(os.path.join(path, "adapter_model.bin")) or
               os.path.exists(os.path.join(path, "adapter_model.safetensors")))
    return has_cfg and has_wts


def _sanity_check(df: pd.DataFrame, src_col: str, out_col: str, flag_col: str = None):
    """Print QA checks that matter before a WMT submission."""
    n = len(df)
    src  = df[src_col].astype(str)
    hyps = df[out_col].astype(str)

    empty_src  = (src.str.strip()  == "").sum()
    empty_hyp  = (hyps.str.strip() == "").sum()
    identical  = (hyps.str.strip() == src.str.strip()).sum()
    avg_src_len = src.str.split().apply(len).mean()
    avg_hyp_len = hyps.str.split().apply(len).mean()

    print(f"\n  ── Sanity checks ──")
    print(f"    Total rows              : {n:,}")
    print(f"    Empty source rows       : {empty_src:,}")
    print(f"    Empty translations      : {empty_hyp:,}")
    print(f"    Translation == source   : {identical:,}")
    print(f"    Avg source length (tok) : {avg_src_len:.1f}")
    print(f"    Avg hyp length (tok)    : {avg_hyp_len:.1f}")

    if empty_hyp > empty_src:
        print(f"    [WARN] {empty_hyp - empty_src} non-empty source rows produced "
              f"EMPTY translations — inspect those rows before submitting.")
    if identical > empty_src:
        print(f"    [WARN] {identical - empty_src} rows have translation == source "
              f"(model may have copied input verbatim) — spot-check these.")
    if empty_hyp == 0 and identical == empty_src:
        print(f"    ✓ No empty or copy-through translations detected.")

    if flag_col and flag_col in df.columns:
        flagged = df[df[flag_col] != ""]
        print(f"    Rows flagged for review : {len(flagged):,}")
        if len(flagged):
            tag_counts = {}
            for tags in flagged[flag_col]:
                for tag in tags.split("|"):
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            for tag, count in sorted(tag_counts.items(), key=lambda kv: -kv[1]):
                preview = flagged.index[flagged[flag_col].str.contains(tag, regex=False)][:10].tolist()
                print(f"      [WARN] {tag:<22} : {count:,}  (rows: {preview})")
            print(f"           See the '{flag_col}' column for the full list.")
        else:
            print(f"    ✓ No script-mixing or degenerate-repetition artefacts detected.")


# ─── Main translation routine ─────────────────────────────────────────────────

def translate_file(
    input_path:  str,
    output_path: str,
    pipeline_id: str,
    adapter_exp: str = "r1",
    text_col:    str = None,
    out_col:     str = "translation",
    num_beams:   int = NUM_BEAMS,
    repetition_penalty: float = REPETITION_PENALTY,
    write_txt:   bool = True,
):
    cfg          = PIPELINE_CONFIGS[pipeline_id]
    src_lang     = cfg["src_lang"]
    tgt_lang     = cfg["tgt_lang"]
    base_name    = get_base_model(pipeline_id)
    adapter_path = adapter_dir(pipeline_id, adapter_exp)

    print(f"\n{'='*64}")
    print(f"  TRANSLATE  Pipeline {pipeline_id}  ({cfg['label']})")
    print(f"  Direction : {src_lang} → {tgt_lang}")
    print(f"  Base      : {base_name}")
    print(f"  Adapter   : {adapter_path}")
    print(f"  Input     : {input_path}")
    print(f"  Num beams : {num_beams}")
    print(f"  Rep. penalty : {repetition_penalty}  (mitigates degenerate repetition loops)")
    print(f"{'='*64}")

    if not _adapter_ready(adapter_path):
        raise FileNotFoundError(
            f"No usable adapter found at {adapter_path}.\n"
            f"Train it first, e.g.:\n"
            f"  python step3_train_dora.py --pipeline {pipeline_id}\n"
            f"or pick a different --adapter that exists for this pipeline."
        )

    df = pd.read_excel(input_path)
    col = _find_text_column(df, text_col)
    print(f"  Text column : '{col}'  ({len(df):,} rows)")

    # ── FIX 1: Strip reference/footnote tags from source before translation ──
    # ── FIX 6: Normalize en-dash/em-dash in numeric ranges ──────────────────
    raw_sentences = [_clean_text(s) for s in df[col].tolist()]
    sentences_no_ref = [_strip_ref_tags(s) for s in raw_sentences]
    sentences = [_normalize_numeric_dashes(s) for s in sentences_no_ref]

    n_ref_stripped = sum(1 for a, b in zip(raw_sentences, sentences_no_ref) if a != b)
    n_dash_fixed   = sum(1 for a, b in zip(sentences_no_ref, sentences) if a != b)
    if n_ref_stripped:
        print(f"  [INFO] Stripped reference/footnote tags from {n_ref_stripped} row(s) "
              f"(e.g. <ID 1>, [1], (1) markers that cause artifact leakage in output).")
    if n_dash_fixed:
        print(f"  [INFO] Normalized en-dash/em-dash in numeric ranges in {n_dash_fixed} row(s) "
              f"(e.g. '2025–26' → '2025-26' to avoid reference-tag artifacts in output).")

    n_blank = sum(1 for s in sentences if not s)
    if n_blank:
        print(f"  [INFO] {n_blank} row(s) have empty/blank source text — "
              f"these will get an empty translation (not sent to the model).")

    print("  Loading tokenizer + base model ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    except Exception:
        print(f"  [WARN] No tokenizer found at {adapter_path} — falling back to the "
              f"base model's tokenizer ({base_name}).")
        tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)

    model_kwargs = {"trust_remote_code": True, "torch_dtype": DTYPE}
    try:
        import flash_attn  # noqa: F401
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("  flash_attn : enabled")
    except ImportError:
        pass

    base = AutoModelForSeq2SeqLM.from_pretrained(base_name, **model_kwargs)

    if len(tokenizer) != base.get_input_embeddings().weight.shape[0]:
        print(f"  [INFO] Resizing token embeddings: {base.get_input_embeddings().weight.shape[0]} "
              f"→ {len(tokenizer)} to match the tokenizer's vocabulary.")
        base.resize_token_embeddings(len(tokenizer))

    print("  Attaching adapter ...")
    model = PeftModel.from_pretrained(base, adapter_path).to(DEVICE)
    model.eval()

    ip = IndicProcessor(inference=True)

    all_preds, all_flags = translate_sentences(
        sentences, model, tokenizer, ip, src_lang, tgt_lang,
        batch_size=INFER_BATCH_SIZE, num_beams=num_beams,
        repetition_penalty=repetition_penalty,
    )

    assert len(all_preds) == len(df) == len(all_flags), (
        f"Row count mismatch: {len(all_preds)} translations vs {len(df)} input rows. "
        f"DO NOT submit — output would be misaligned with the source file."
    )

    flag_col = f"{out_col}_flag"
    df[out_col]  = all_preds
    df[flag_col] = all_flags

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"\n  ✓ Saved → {output_path}")

    if write_txt:
        txt_path = os.path.splitext(output_path)[0] + ".txt"
        with open(txt_path, "w", encoding="utf-8", newline="\n") as f:
            for t in df[out_col].astype(str).tolist():
                f.write(t + "\n")
        print(f"  ✓ Plain-text (WMT submission format, {len(df):,} lines, "
              f"UTF-8, one per line) → {txt_path}")

    _sanity_check(df, col, out_col, flag_col=flag_col)

    del model, base, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return df


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--pipeline", required=True, choices=ALL_PIPELINES,
        help="Which pipeline's base model + adapter to use "
             "(A: eng→mni_Mtei, B: mni_Mtei→eng, C: eng→mni_Beng, D: mni_Beng→eng)",
    )
    parser.add_argument("--input",  required=True, help="Path to input .xlsx")
    parser.add_argument("--output", required=True, help="Path to output .xlsx")
    parser.add_argument(
        "--adapter", default="r1",
        help="Adapter experiment dir to load: r1, exp1_bt, exp2_bt_ft, "
             "exp3_bt_ft_iter, exp4_noise, exp5_subword_reg, "
             "exp6_bt_noise, exp7_bt_subword, ... (default: r1)",
    )
    parser.add_argument(
        "--text-col", default=None,
        help="Name of the column containing source text. "
             "Auto-detected if not given (tries 'src_text', 'source', "
             "'text', 'sentence', 'input', else the first column).",
    )
    parser.add_argument(
        "--out-col", default="translation",
        help="Name of the new column to write translations into "
             "(default: 'translation').",
    )
    parser.add_argument(
        "--num-beams", type=int, default=NUM_BEAMS,
        help=f"Beam size for generation (default: {NUM_BEAMS}, from config.py).",
    )
    parser.add_argument(
        "--repetition-penalty", type=float, default=REPETITION_PENALTY,
        help=f"Soft per-token repetition penalty (default: {REPETITION_PENALTY}). "
             f"1.0 = off; 1.3-1.6 is a reasonable range.",
    )
    parser.add_argument(
        "--no-txt", action="store_true",
        help="Do not also write a plain-text (.txt) file with one translation "
             "per line.",
    )
    args = parser.parse_args()

    translate_file(
        input_path  = args.input,
        output_path = args.output,
        pipeline_id = args.pipeline,
        adapter_exp = args.adapter,
        text_col    = args.text_col,
        out_col     = args.out_col,
        num_beams   = args.num_beams,
        repetition_penalty = args.repetition_penalty,
        write_txt   = not args.no_txt,
    )


if __name__ == "__main__":
    main()
