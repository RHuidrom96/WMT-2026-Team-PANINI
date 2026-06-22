"""
step1_analysis.py — Dataset Validation, Cleaning & Corpus Analysis

"""

import os
import re
import argparse
import unicodedata
import warnings
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter

warnings.filterwarnings("ignore")

from config import (
    OUTPUT_DIR, FIGURES_DIR, SEED,
    ALL_PIPELINES, PIPELINE_CONFIGS,
    raw_path, split_path,
)

random.seed(SEED)
np.random.seed(SEED)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
plt.rcParams.update({
    "figure.dpi":   150,
    "savefig.dpi":  300,
    "savefig.bbox": "tight",
    "font.family":  "DejaVu Sans",
})

PAL = {
    "src":   "#4C72B0",
    "tgt":   "#DD8452",
    "clean": "#55A868",
    "raw":   "#C44E52",
    "train": "#4C72B0",
    "dev":   "#55A868",
    "test":  "#8172B2",
}

# Token-length filter bounds
MIN_SRC_TOKENS = 2
MAX_SRC_TOKENS = 200
MIN_TGT_TOKENS = 1
MAX_TGT_TOKENS = 200

# ── Column-name aliases (direction-AWARE — built per pipeline at load time) ────

# Column names that always identify English text
_ENG_COL_NAMES = {
    "eng_src", "english", "english_src", "english_tgt",
    "eng", "en", "source_en", "target_en",
}

# Column names that always identify Meitei/Manipuri text (either script)
_MNI_COL_NAMES = {
    "meti_tgt", "meitei", "meitei_tgt", "meetei", "meetei_tgt",
    "mni_mtei", "mni_mtei_tgt", "mni_mtei_src",
    "mni_beng", "mni_beng_tgt", "mni_beng_src",
    "mni_tgt", "mni_src", "mni", "manipuri",
}

# Direction-neutral aliases (position/role names like "source", "target")
_NEUTRAL_ALIASES = {
    "source":      "src_text",
    "src":         "src_text",
    "input":       "src_text",
    "target":      "tgt_text",
    "tgt":         "tgt_text",
    "output":      "tgt_text",
    "translation": "tgt_text",
}

# Unicode block heuristic: does this text look like Meitei Mayek?
# Bengali block (for mni_Beng): U+0980–U+09FF
_MEITEI_RANGES = [(0xAAE0, 0xAAF6), (0xABC0, 0xABED)]
_BENGALI_RANGE = (0x0980, 0x09FF)


def _meitei_char_ratio(series: pd.Series, sample: int = 500) -> float:
    """Return fraction of sampled characters that fall in Meitei/Bengali blocks."""
    texts = series.dropna().astype(str).head(sample)
    total = meitei = 0
    for text in texts:
        for ch in text:
            cp = ord(ch)
            if ch.isalpha():
                total += 1
                if any(lo <= cp <= hi for lo, hi in _MEITEI_RANGES) or \
                   _BENGALI_RANGE[0] <= cp <= _BENGALI_RANGE[1]:
                    meitei += 1
    return meitei / max(total, 1)

# ─── Source-text preprocessing (mirrors translate_inference.py) ───────────────


def _clean_text(text) -> str:

    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r" *꯭ *", "꯭", text)
    return text


_REF_TAG_RE = re.compile(
    r'<[^>]{1,30}?\s+\d+\s*>'   # <word 1>, <ꯑꯥꯏ ꯗꯤ ꯱>, etc.
    r'|\[\d+\]'                  # [1], [23]
    r'|\(\d+\)',                 # (1), (23)
    re.UNICODE,
)

def _strip_ref_tags(text: str) -> str:
    """
    Remove reference/footnote markers (e.g. <ID 1>, [1], (1)) from text.
    These cause reference-tag artifacts in model output when left in training data.
    Mirrors translate_inference.py::_strip_ref_tags().
    """
    return re.sub(r'\s+', ' ', _REF_TAG_RE.sub('', text)).strip()


_NUMERIC_DASH_RE = re.compile(r'(\d)\s*[–—]\s*(\d)')  # en-dash U+2013, em-dash U+2014

def _normalize_numeric_dashes(text: str) -> str:

    return _NUMERIC_DASH_RE.sub(r'\1-\2', text)


def _apply_inference_preprocessing(series: pd.Series) -> pd.Series:
    """Apply all three inference-side source preprocessing steps to a Series."""
    return (
        series
        .apply(_clean_text)
        .apply(_strip_ref_tags)
        .apply(_normalize_numeric_dashes)
    )


# ─── Transliteration / script-normalization (mirrors translate_inference.py) ──


_ASCII_TO_MTEI = str.maketrans("0123456789", "꯰꯱꯲꯳꯴꯵꯶꯷꯸꯹")
_ASCII_TO_BENG = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")

def _fix_digits(text: str, lang: str) -> str:
    """Convert ASCII digits to the native script digit form for Meitei columns."""
    if lang == "mni_Mtei":
        return text.translate(_ASCII_TO_MTEI)
    elif lang == "mni_Beng":
        return text.translate(_ASCII_TO_BENG)
    return text   # eng_Latn — leave digits as ASCII


_ENG_LETTER_MTEI = {
    'A': 'ꯑꯦ',  'B': 'ꯕꯤ',   'C': 'ꯁꯤ',      'D': 'ꯗꯤ',   'E': 'ꯏ',
    'F': 'ꯐ',   'G': 'ꯖꯤ',   'H': 'ꯦꯏꯆ',    'I': 'ꯑꯥꯏ', 'J': 'ꯖꯦ',
    'K': 'ꯀꯦ',  'L': 'ꯑꯦꯜ', 'M': 'ꯑꯦꯝ',    'N': 'ꯑꯦꯟ', 'O': 'ꯑꯣ',
    'P': 'ꯄꯤ',  'Q': 'ꯀ꭛ꯌꯨ','R': 'ꯑꯥꯔ',    'S': 'ꯑꯦꯁ', 'T': 'ꯇꯤ',
    'U': 'ꯌꯨ',  'V': 'ꯚꯤ',   'W': 'ꯗꯕ꭛ꯂꯌꯨ','X': 'ꯑꯦꯛꯁ','Y': 'ꯋꯥꯏ',
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

def _transliterate_english_chars(text: str, lang: str) -> str:
    """
    Convert residual ASCII English letters in a Meitei column to their
    spoken-name equivalents in the target script.
    No-op for eng_Latn columns.
    """
    if lang == "mni_Mtei":
        letter_map = _ENG_LETTER_MTEI
    elif lang == "mni_Beng":
        letter_map = _ENG_LETTER_BENG
    else:
        return text   # eng_Latn — leave as-is

    if not re.search(r'[A-Za-z]', text):
        return text   # fast-path: nothing to do

    def _replace(m: re.Match) -> str:
        return ''.join(letter_map.get(ch.upper(), ch) for ch in m.group(0))

    return re.sub(r'[A-Za-z]+', _replace, text)


# Devanagari letter range (excludes shared punctuation like danda ।)
_DEVANAGARI_LETTER_RE = re.compile(
    r'[\u0904-\u0963\u0966-\u096F\u0971-\u097F]', re.UNICODE,
)
_DEVANAGARI_RUN_RE = re.compile(
    r'[\u0904-\u0963\u0966-\u096F\u0971-\u097F]'
    r'[\u0900-\u0963\u0966-\u097F\u0902\u0903\u093C\u093E-\u094C\u094D\u0951-\u0954]*',
    re.UNICODE,
)

def _strip_devanagari_from_meitei(text: str) -> str:
    """Remove Devanagari letter fragments leaked into Meitei training targets."""
    cleaned = _DEVANAGARI_RUN_RE.sub('', text)
    return re.sub(r'\s+', ' ', cleaned).strip()


# Non-Latin script regex for stripping from English columns
_NON_LATIN_SCRIPT_RE = re.compile(
    r'(?:[\u0900-\u097F\u0980-\u09FF\uAAE0-\uAAFF\uABC0-\uABFF]'
    r'[\u0900-\u097F\u0980-\u09FF\uAAE0-\uAAFF\uABC0-\uABFF\u09CD\u0ACD\u0BCD.'
    r'\u200C\u200D]*)+',
    re.UNICODE,
)

def _strip_source_script_from_english(text: str) -> str:
    """Remove any Brahmic/Meitei script fragments leaked into an English column."""
    cleaned = _NON_LATIN_SCRIPT_RE.sub('', text)
    return re.sub(r'\s+', ' ', cleaned).strip()


_CONSEC_DUP_RE = re.compile(r'\b(\S+)( \1)+\b', re.IGNORECASE)

def _remove_consecutive_duplicates(text: str) -> str:

    return _CONSEC_DUP_RE.sub(r'\1', text)


def _capitalize_first(text: str) -> str:

    for i, ch in enumerate(text):
        if ch.isalpha():
            if ch.islower():
                return text[:i] + ch.upper() + text[i + 1:]
            return text
    return text


def _apply_script_normalization(series: pd.Series, lang: str) -> pd.Series:

    if lang in ("mni_Mtei", "mni_Beng"):
        return (
            series
            .apply(lambda t: _fix_digits(t, lang))
            .apply(lambda t: _strip_devanagari_from_meitei(t))
            .apply(lambda t: _transliterate_english_chars(t, lang))
        )
    elif lang == "eng_Latn":
        return (
            series
            .apply(_strip_source_script_from_english)
            .apply(_remove_consecutive_duplicates)
            .apply(_capitalize_first)
        )
    return series   # unknown lang — pass through unchanged


SPLITS = ["train", "dev", "test"]


# ─── Figure helpers ───────────────────────────────────────────────────────────

def savefig(fig_dir: str, name: str):
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {name}")


def tok_len(series: pd.Series) -> pd.Series:
    return series.dropna().astype(str).apply(lambda x: len(x.split()))


def char_len(series: pd.Series) -> pd.Series:
    return series.dropna().astype(str).apply(len)


def vocab_stats(series: pd.Series, name: str) -> dict:
    tokens = [t for sent in series.dropna().astype(str) for t in sent.split()]
    vocab  = set(tokens)
    return {
        "name":  name,
        "total": len(tokens),
        "vocab": len(vocab),
        "ttr":   round(len(vocab) / max(len(tokens), 1), 4),
        "top20": Counter(tokens).most_common(20),
    }


# ─── 1. Load & column normalisation ──────────────────────────────────────────

def load_split(pipeline_id: str, split: str) -> pd.DataFrame:
    path = raw_path(pipeline_id, split)
    print(f"\n  Loading pipeline {pipeline_id} [{split}]: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            f"Place pipeline_{pipeline_id}_{split}.xlsx under ./data/\n"
            f"  (or run the rename helper in the README)"
        )
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    print(f"  Raw shape  : {df.shape}")
    print(f"  Columns    : {df.columns.tolist()}")

    # ── Direction-aware column renaming ───────────────────────────────────────

    cfg         = PIPELINE_CONFIGS[pipeline_id]
    en2indic    = cfg["direction"] == "en2indic"   # True for A/C, False for B/D

    rename_map = {}
    cols_lower = {c.lower().strip(): c for c in df.columns}

    for col_lower, col_orig in cols_lower.items():
        if col_orig in ("src_text", "tgt_text"):
            continue                        # already canonical — skip

        if col_lower in _ENG_COL_NAMES:
            target = "src_text" if en2indic else "tgt_text"
            if target not in df.columns:
                rename_map[col_orig] = target

        elif col_lower in _MNI_COL_NAMES:
            target = "tgt_text" if en2indic else "src_text"
            if target not in df.columns:
                rename_map[col_orig] = target

        elif col_lower in _NEUTRAL_ALIASES:
            canonical = _NEUTRAL_ALIASES[col_lower]
            if canonical not in df.columns:
                rename_map[col_orig] = canonical

    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"  Renamed    : {rename_map}")

    required = {"src_text", "tgt_text"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Pipeline {pipeline_id} [{split}]: missing column(s) {missing}.\n"
            f"Available columns: {df.columns.tolist()}\n"
            f"Rename your columns to 'src_text' and 'tgt_text', or add an entry\n"
            f"to the alias sets (_ENG_COL_NAMES / _MNI_COL_NAMES) in step1_analysis.py."
        )

    # ── Content-based sanity check: detect if src/tgt are swapped ────────────

    src_meitei = _meitei_char_ratio(df["src_text"])
    tgt_meitei = _meitei_char_ratio(df["tgt_text"])


    src_looks_meitei = src_meitei > 0.5
    tgt_looks_meitei = tgt_meitei > 0.5

    swap_needed = False
    if en2indic and src_looks_meitei and not tgt_looks_meitei:
        swap_needed = True
    elif not en2indic and not src_looks_meitei and tgt_looks_meitei:
        swap_needed = True

    if swap_needed:
        print(
            f"  ⚠ CONTENT MISMATCH DETECTED for pipeline {pipeline_id} [{split}]:\n"
            f"     src Meitei-ratio={src_meitei:.2f}, tgt Meitei-ratio={tgt_meitei:.2f}\n"
            f"     Expected direction: {'en→Indic' if en2indic else 'Indic→en'}\n"
            f"     → AUTO-SWAPPING src_text ↔ tgt_text to correct column assignment."
        )
        df = df.rename(columns={"src_text": "_tmp_src", "tgt_text": "_tmp_tgt"})
        df = df.rename(columns={"_tmp_src": "tgt_text", "_tmp_tgt": "src_text"})
    else:
        print(
            f"  ✓ Content check OK  "
            f"(src Meitei-ratio={src_meitei:.2f}, tgt Meitei-ratio={tgt_meitei:.2f})"
        )

    return df


# ─── 2. Corpus analysis (rich — mirrors original step1) ──────────────────────

def analyse(df: pd.DataFrame, tag: str, split_name: str,
            pipeline_id: str, cfg: dict, fig_dir: str):
    """
    Full corpus analysis on df (raw or clean).
    Produces: stats printout, CSV, 5 figure files.
    """
    src_col = "src_text"
    tgt_col = "tgt_text"

    src_label = cfg["src_lang"]
    tgt_label = cfg["tgt_lang"]

    print(f"\n{'='*60}")
    print(f"  CORPUS ANALYSIS [{tag.upper()}]  [pipeline {pipeline_id}]"
          f"  [{split_name}]")
    print(f"{'='*60}")
    print(f"  Pairs            : {len(df):,}")
    print(f"  Missing src      : {df[src_col].isna().sum()}")
    print(f"  Missing tgt      : {df[tgt_col].isna().sum()}")

    src_l = tok_len(df[src_col])
    tgt_l = tok_len(df[tgt_col])
    src_c = char_len(df[src_col])
    tgt_c = char_len(df[tgt_col])
    src_v = vocab_stats(df[src_col], src_label)
    tgt_v = vocab_stats(df[tgt_col], tgt_label)

    # ── Token-length stats table ───────────────────────────────────────────────
    rows = []
    for lens, name in [(src_l, f"{src_label} (src)"),
                       (tgt_l, f"{tgt_label} (tgt)")]:
        rows.append({
            "Language":     name,
            "Sentences":    len(lens),
            "Total Tokens": int(lens.sum()),
            "Mean":         round(float(lens.mean()), 2),
            "Median":       int(lens.median()),
            "Min":          int(lens.min()),
            "Max":          int(lens.max()),
            "Std":          round(float(lens.std()), 2),
        })
    stats_df = pd.DataFrame(rows).set_index("Language")
    print(f"\n  Token-length statistics:")
    print(stats_df.to_string())

    stats_csv = os.path.join(
        OUTPUT_DIR, f"pipeline_{pipeline_id}",
        f"corpus_statistics_{tag}_{split_name}.csv",
    )
    stats_df.to_csv(stats_csv)

    # ── Vocabulary / TTR ───────────────────────────────────────────────────────
    print(f"\n  Vocabulary & Type-Token Ratio:")
    for v in [src_v, tgt_v]:
        print(f"    {v['name']:30s}: tokens={v['total']:,}  "
              f"vocab={v['vocab']:,}  TTR={v['ttr']:.4f}")

    # ── Coverage percentiles ───────────────────────────────────────────────────
    print(f"\n  Coverage percentiles (token length):")
    for pct in [75, 90, 95, 99]:
        s_pct = int(np.percentile(src_l, pct))
        t_pct = int(np.percentile(tgt_l, pct))
        print(f"    {pct}%  {src_label} ≤ {s_pct} tok"
              f"  |  {tgt_label} ≤ {t_pct} tok")

    # ── Top-20 tokens ──────────────────────────────────────────────────────────
    print(f"\n  Top-10 source tokens: "
          f"{[w for w,_ in src_v['top20'][:10]]}")
    print(f"  Top-10 target tokens: "
          f"{[w for w,_ in tgt_v['top20'][:10]]}")

    px = f"{pipeline_id}_{split_name}_{tag}_"

    # ── Figure 1: Token-length histograms (src + tgt) ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (lens, label, col) in zip(axes, [
            (src_l, f"{src_label} (Source)", PAL["src"]),
            (tgt_l, f"{tgt_label} (Target)", PAL["tgt"])]):
        ax.hist(lens, bins=50, color=col, edgecolor="white", alpha=0.85)
        ax.axvline(float(lens.mean()),   color="red",   linestyle="--", lw=1.5,
                   label=f"Mean={lens.mean():.1f}")
        ax.axvline(float(lens.median()), color="green", linestyle=":",  lw=1.5,
                   label=f"Median={int(lens.median())}")
        ax.set_title(f"Token Length — {label}\n[pipeline {pipeline_id} | {split_name} | {tag}]",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("# Tokens")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=9)
    plt.suptitle(
        f"Sentence-Length Distribution  "
        f"[Pipeline {pipeline_id} | {split_name} | {tag}]",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    savefig(fig_dir, f"{px}01_length_dist.png")

    # ── Figure 2: Target/Source length ratio ──────────────────────────────────
    mask   = df[src_col].notna() & df[tgt_col].notna()
    ratios = (
        df.loc[mask, tgt_col].astype(str).apply(lambda x: len(x.split())) /
        df.loc[mask, src_col].astype(str).apply(lambda x: max(1, len(x.split())))
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(ratios, bins=60, color=PAL["tgt"], edgecolor="white", alpha=0.85)
    ax.axvline(float(ratios.mean()), color="red", linestyle="--", lw=1.5,
               label=f"Mean={ratios.mean():.2f}")
    ax.axvline(float(ratios.median()), color="green", linestyle=":", lw=1.5,
               label=f"Median={ratios.median():.2f}")
    ax.set_title(
        f"Target / Source Token-Length Ratio\n"
        f"[Pipeline {pipeline_id} | {split_name} | {tag}]",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Length Ratio (tgt tokens / src tokens)")
    ax.set_ylabel("Frequency")
    ax.legend()
    print(f"  Length ratio  mean={ratios.mean():.3f}  "
          f"median={ratios.median():.3f}  "
          f"std={ratios.std():.3f}  "
          f"(>3: {(ratios>3).sum()}, <0.33: {(ratios<0.33).sum()})")
    plt.tight_layout()
    savefig(fig_dir, f"{px}02_length_ratio.png")

    # ── Figure 3: Character-length distributions ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (clens, label, col) in zip(axes, [
            (src_c, f"{src_label} (Source)", PAL["src"]),
            (tgt_c, f"{tgt_label} (Target)", PAL["tgt"])]):
        ax.hist(clens, bins=50, color=col, edgecolor="white", alpha=0.85)
        ax.axvline(float(clens.mean()),   color="red",   linestyle="--", lw=1.5,
                   label=f"Mean={clens.mean():.0f}")
        ax.axvline(float(clens.median()), color="green", linestyle=":",  lw=1.5,
                   label=f"Median={int(clens.median())}")
        ax.set_title(f"Character Length — {label}\n"
                     f"[pipeline {pipeline_id} | {split_name} | {tag}]",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("# Characters")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=9)
    plt.suptitle(
        f"Character-Length Distribution  "
        f"[Pipeline {pipeline_id} | {split_name} | {tag}]",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    savefig(fig_dir, f"{px}03_char_dist.png")

    # ── Figure 4: Vocabulary rank-frequency (Zipf) ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (v, label, col) in zip(axes, [
            (src_v, f"{src_label} (Source)", PAL["src"]),
            (tgt_v, f"{tgt_label} (Target)", PAL["tgt"])]):
        freqs = [f for _, f in v["top20"]]
        words = [w for w, _ in v["top20"]]
        ax.bar(range(len(freqs)), freqs, color=col, edgecolor="white", alpha=0.85)
        ax.set_xticks(range(len(words)))
        ax.set_xticklabels(words, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"Top-20 Tokens — {label}\n"
                     f"[pipeline {pipeline_id} | {split_name} | {tag}]",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Token")
        ax.set_ylabel("Frequency")
    plt.tight_layout()
    savefig(fig_dir, f"{px}04_top20_tokens.png")

    # ── Figure 5: Scatter — src length vs tgt length ─────────────────────────
    sample_n = min(2000, len(src_l))
    idx      = np.random.choice(len(src_l), sample_n, replace=False)
    src_samp = src_l.iloc[idx]
    tgt_samp = tgt_l.iloc[idx]
    fig, ax  = plt.subplots(figsize=(7, 6))
    ax.scatter(src_samp, tgt_samp, alpha=0.25, s=8, color=PAL["tgt"])
    # Trend line
    z = np.polyfit(src_samp, tgt_samp, 1)
    p = np.poly1d(z)
    xs = np.linspace(src_samp.min(), src_samp.max(), 100)
    ax.plot(xs, p(xs), color="red", lw=1.5, label=f"y={z[0]:.2f}x+{z[1]:.1f}")
    ax.set_title(
        f"Src vs Tgt Token Length  (n={sample_n:,})\n"
        f"[Pipeline {pipeline_id} | {split_name} | {tag}]",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel(f"Source ({src_label}) tokens")
    ax.set_ylabel(f"Target ({tgt_label}) tokens")
    ax.legend(fontsize=9)
    plt.tight_layout()
    savefig(fig_dir, f"{px}05_src_vs_tgt_scatter.png")

    return src_v, tgt_v


# ─── 3. Cleaning ──────────────────────────────────────────────────────────────

def clean(df_raw: pd.DataFrame, split_name: str, pipeline_id: str) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"  CLEANING & DEDUPLICATION  "
          f"[pipeline {pipeline_id}]  [{split_name}]")
    print(f"{'='*60}")
    df = df_raw.copy()
    n0 = len(df)
    print(f"  Before         : {n0:,}")

    # Drop NaN
    df = df.dropna(subset=["src_text", "tgt_text"])
    print(f"  After drop NaN : {len(df):,}  (−{n0-len(df):,})")

    # Strip whitespace on all object columns
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].astype(str).str.strip()

    # ── Inference-mirrored preprocessing (FIX 1, FIX 6, _clean_text) ─────────

    for col in ("src_text", "tgt_text"):
        original = df[col].copy()
        df[col] = _apply_inference_preprocessing(df[col])
        n_changed = (df[col] != original).sum()
        if n_changed:
            # Count which fixes actually fired
            n_ref   = (original.apply(_strip_ref_tags) != original).sum()
            n_dash  = (original.apply(_normalize_numeric_dashes) != original).sum()
            n_norm  = (original.apply(_clean_text) != original).sum()
            print(
                f"  Preproc [{col}]  {n_changed:,} row(s) changed  "
                f"(NFC/whitespace: {n_norm}, ref-tags: {n_ref}, "
                f"numeric-dashes: {n_dash})"
            )

    # ── Script-specific normalization (transliteration) ──────────────────────
    cfg_now   = PIPELINE_CONFIGS[pipeline_id]
    src_lang  = cfg_now["src_lang"]   # e.g. "eng_Latn" for A/C, "mni_Mtei" for B
    tgt_lang  = cfg_now["tgt_lang"]   # e.g. "mni_Mtei" for A, "eng_Latn" for B/D

    for col, lang in (("src_text", src_lang), ("tgt_text", tgt_lang)):
        original  = df[col].copy()
        df[col]   = _apply_script_normalization(df[col], lang)
        n_changed = (df[col] != original).sum()
        if n_changed:
            # Break down by which sub-step fired
            if lang in ("mni_Mtei", "mni_Beng"):
                n_digits = (original.apply(lambda t: _fix_digits(t, lang))
                            != original).sum()
                n_deva   = (original.apply(_strip_devanagari_from_meitei)
                            != original).sum()
                n_eng    = (original.apply(lambda t: _transliterate_english_chars(t, lang))
                            != original).sum()
                print(
                    f"  Script norm [{col} / {lang}]  {n_changed:,} row(s) changed  "
                    f"(digits: {n_digits}, devanagari-strip: {n_deva}, "
                    f"eng-transliterate: {n_eng})"
                )
            elif lang == "eng_Latn":
                n_strip = (original.apply(_strip_source_script_from_english)
                           != original).sum()
                n_dup   = (original.apply(_remove_consecutive_duplicates)
                           != original).sum()
                n_cap   = (original.apply(_capitalize_first)
                           != original).sum()
                print(
                    f"  Script norm [{col} / {lang}]  {n_changed:,} row(s) changed  "
                    f"(script-strip: {n_strip}, consec-dups: {n_dup}, "
                    f"capitalize: {n_cap})"
                )

    # Drop empty / "nan" strings
    b = len(df)
    df = df[
        df["src_text"].ne("") & df["tgt_text"].ne("") &
        df["src_text"].ne("nan") & df["tgt_text"].ne("nan")
    ]
    print(f"  After drop empty: {len(df):,}  (−{b-len(df):,})")

    # Drop identical src == tgt (copy-through rows)
    b = len(df)
    df = df[df["src_text"].ne(df["tgt_text"])]
    print(f"  After drop copy : {len(df):,}  (−{b-len(df):,})")

    # Deduplicate within-split on (src_text, tgt_text)
    b = len(df)

    print(f"  After dedup     : {len(df):,}  (−{b-len(df):,})")

    # Token-length filter
    src_tok = df["src_text"].astype(str).apply(lambda x: len(x.split()))
    tgt_tok = df["tgt_text"].astype(str).apply(lambda x: len(x.split()))
    mask    = (
        (src_tok >= MIN_SRC_TOKENS) & (src_tok <= MAX_SRC_TOKENS) &
        (tgt_tok >= MIN_TGT_TOKENS) & (tgt_tok <= MAX_TGT_TOKENS)
    )
    b   = len(df)
    df  = df[mask]
    print(f"  After length flt: {len(df):,}  (−{b-len(df):,})"
          f"  [{MIN_SRC_TOKENS}≤src≤{MAX_SRC_TOKENS},"
          f" {MIN_TGT_TOKENS}≤tgt≤{MAX_TGT_TOKENS} tokens]")

    # Metadata columns
    cfg            = PIPELINE_CONFIGS[pipeline_id]
    df["pipeline"] = pipeline_id
    df["split"]    = split_name
    df["src_lang"] = cfg["src_lang"]
    df["tgt_lang"] = cfg["tgt_lang"]

    df = df.reset_index(drop=True)
    print(f"  ✓ Clean final   : {len(df):,} pairs  [pipeline {pipeline_id} | {split_name}]")
    return df


# ─── 4. Leak check ────────────────────────────────────────────────────────────

def leak_check(train_df: pd.DataFrame,
               dev_df:   pd.DataFrame,
               test_df:  pd.DataFrame,
               pipeline_id: str):
    print(f"\n  {'─'*50}")
    print(f"  LEAK CHECK  [pipeline {pipeline_id}]")
    print(f"  {'─'*50}")

    train_pairs = set(zip(train_df["src_text"], train_df["tgt_text"]))
    dev_pairs   = set(zip(dev_df["src_text"],   dev_df["tgt_text"]))
    test_pairs  = set(zip(test_df["src_text"],  test_df["tgt_text"]))

    leak_dev  = len(train_pairs & dev_pairs)
    leak_test = len(train_pairs & test_pairs)
    dev_test  = len(dev_pairs  & test_pairs)

    if leak_dev:
        print(f"  ⚠ WARNING : {leak_dev:,} train↔dev overlapping pairs!")
    else:
        print(f"  ✓ No train↔dev overlap.")

    if leak_test:
        print(f"  ⚠ WARNING : {leak_test:,} train↔test overlapping pairs!")
    else:
        print(f"  ✓ No train↔test overlap.")

    if dev_test:
        print(f"  ⚠ WARNING : {dev_test:,} dev↔test overlapping pairs!")
    else:
        print(f"  ✓ No dev↔test overlap.")

    return {"train_dev": leak_dev, "train_test": leak_test, "dev_test": dev_test}


# ─── 5. Summary figures ───────────────────────────────────────────────────────

def plot_split_summary(splits: dict, pipeline_id: str, fig_dir: str):
    """Bar chart: raw vs clean pair counts for train / dev / test."""
    labels = list(splits.keys())
    raw_v  = [splits[s]["raw"]   for s in labels]
    cln_v  = [splits[s]["clean"] for s in labels]

    x   = np.arange(len(labels))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1  = ax.bar(x - w/2, raw_v,  w, label="Raw",   color=PAL["raw"],   alpha=0.85,
                 edgecolor="white")
    b2  = ax.bar(x + w/2, cln_v,  w, label="Clean", color=PAL["clean"], alpha=0.85,
                 edgecolor="white")
    ax.bar_label(b1, fmt="%d", padding=3, fontsize=9, fontweight="bold")
    ax.bar_label(b2, fmt="%d", padding=3, fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_title(
        f"Dataset Split Sizes — Raw vs Clean\n[Pipeline {pipeline_id}  "
        f"({PIPELINE_CONFIGS[pipeline_id]['label']})]",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylabel("Sentence Pairs")
    ax.set_ylim(0, max(max(raw_v), max(cln_v)) * 1.18)
    ax.legend()
    plt.tight_layout()
    savefig(fig_dir, "split_summary.png")


def plot_combined_length_comparison(splits_clean: dict,
                                    pipeline_id: str, fig_dir: str):
    """Overlapping length histograms for all three splits on one canvas."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"train": PAL["train"], "dev": PAL["dev"], "test": PAL["test"]}

    for ax, side in zip(axes, ["src_text", "tgt_text"]):
        for sp, df in splits_clean.items():
            lens = tok_len(df[side])
            ax.hist(lens, bins=40, alpha=0.55, color=colors[sp],
                    label=f"{sp} (n={len(lens):,})", density=True)
        col_label = ("Source" if side == "src_text" else "Target")
        cfg = PIPELINE_CONFIGS[pipeline_id]
        lang = cfg["src_lang"] if side == "src_text" else cfg["tgt_lang"]
        ax.set_title(f"{col_label} ({lang}) Token Length\n"
                     f"[Pipeline {pipeline_id} — all splits]",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("# Tokens")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
    plt.suptitle(
        f"Token-Length Distribution Across Splits  [Pipeline {pipeline_id}]",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    savefig(fig_dir, "all_splits_length_overlay.png")


# ─── 6. Per-pipeline corpus stats CSV ────────────────────────────────────────

def save_corpus_stats(splits_clean: dict, pipeline_id: str):
    cfg  = PIPELINE_CONFIGS[pipeline_id]
    rows = []
    for sp, df in splits_clean.items():
        src_l = tok_len(df["src_text"])
        tgt_l = tok_len(df["tgt_text"])
        src_c = char_len(df["src_text"])
        tgt_c = char_len(df["tgt_text"])
        rows.append({
            "pipeline":        pipeline_id,
            "split":           sp,
            "src_lang":        cfg["src_lang"],
            "tgt_lang":        cfg["tgt_lang"],
            "n_pairs":         len(df),
            "src_mean_tok":    round(float(src_l.mean()), 2),
            "src_median_tok":  int(src_l.median()),
            "src_max_tok":     int(src_l.max()),
            "tgt_mean_tok":    round(float(tgt_l.mean()), 2),
            "tgt_median_tok":  int(tgt_l.median()),
            "tgt_max_tok":     int(tgt_l.max()),
            "src_mean_char":   round(float(src_c.mean()), 2),
            "tgt_mean_char":   round(float(tgt_c.mean()), 2),
            "src_vocab":       len({t for s in df["src_text"]
                                    for t in str(s).split()}),
            "tgt_vocab":       len({t for s in df["tgt_text"]
                                    for t in str(s).split()}),
            "len_ratio_mean":  round(float(
                (tgt_l / src_l.clip(lower=1)).mean()), 3),
        })
    stats_df = pd.DataFrame(rows)
    out = os.path.join(OUTPUT_DIR, f"pipeline_{pipeline_id}",
                       "corpus_statistics_summary.csv")
    stats_df.to_csv(out, index=False)
    print(f"\n  ✓ Summary stats → {out}")
    print(stats_df.to_string(index=False))
    return stats_df


# ─── 7. Per-pipeline processing ───────────────────────────────────────────────

def process_pipeline(pipeline_id: str):
    cfg     = PIPELINE_CONFIGS[pipeline_id]
    fig_dir = os.path.join(FIGURES_DIR, f"pipeline_{pipeline_id}")
    os.makedirs(fig_dir, exist_ok=True)
    out_dir = os.path.join(OUTPUT_DIR, f"pipeline_{pipeline_id}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  PIPELINE {pipeline_id}  —  {cfg['label']}")
    print(f"  {cfg['src_lang']}  →  {cfg['tgt_lang']}")
    print(f"{'='*64}")

    # ── Load all three splits ─────────────────────────────────────────────────
    raw = {}
    for sp in SPLITS:
        raw[sp] = load_split(pipeline_id, sp)

    # ── Corpus analysis: RAW ──────────────────────────────────────────────────
    print(f"\n{'─'*64}")
    print(f"  RAW DATA ANALYSIS  [pipeline {pipeline_id}]")
    print(f"{'─'*64}")
    for sp in SPLITS:
        analyse(raw[sp], tag="raw", split_name=sp,
                pipeline_id=pipeline_id, cfg=cfg, fig_dir=fig_dir)

    # ── Clean each split ──────────────────────────────────────────────────────
    print(f"\n{'─'*64}")
    print(f"  CLEANING  [pipeline {pipeline_id}]")
    print(f"{'─'*64}")
    clean_splits = {}
    for sp in SPLITS:
        clean_splits[sp] = clean(raw[sp], sp, pipeline_id)

    # ── Corpus analysis: CLEAN ────────────────────────────────────────────────
    print(f"\n{'─'*64}")
    print(f"  CLEAN DATA ANALYSIS  [pipeline {pipeline_id}]")
    print(f"{'─'*64}")
    for sp in SPLITS:
        analyse(clean_splits[sp], tag="clean", split_name=sp,
                pipeline_id=pipeline_id, cfg=cfg, fig_dir=fig_dir)

    # ── Leak check ────────────────────────────────────────────────────────────
    leaks = leak_check(
        clean_splits["train"],
        clean_splits["dev"],
        clean_splits["test"],
        pipeline_id,
    )

    # ── Save cleaned splits ────────────────────────────────────────────────────
    for sp, df in clean_splits.items():
        out = split_path(pipeline_id, sp)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        df.to_excel(out, index=False)
        print(f"  ✓ Saved {sp:5s} → {out}  ({len(df):,} rows)")

    # ── Summary figures ───────────────────────────────────────────────────────
    plot_split_summary(
        {sp: {"raw": len(raw[sp]), "clean": len(clean_splits[sp])}
         for sp in SPLITS},
        pipeline_id, fig_dir,
    )
    plot_combined_length_comparison(clean_splits, pipeline_id, fig_dir)

    # ── Global corpus stats CSV ───────────────────────────────────────────────
    save_corpus_stats(clean_splits, pipeline_id)

    return clean_splits, leaks


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pipeline", nargs="+", choices=ALL_PIPELINES, default=ALL_PIPELINES,
        metavar="ID",
        help="Pipeline(s) to process. Default: all (A B C D).",
    )
    args = parser.parse_args()

    print("\n" + "="*64)
    print("  STEP 1 — DATASET VALIDATION, CLEANING & CORPUS ANALYSIS")
    print(f"  Pipelines : {args.pipeline}")
    print("="*64)

    all_results = {}
    for pid in args.pipeline:
        clean_splits, leaks = process_pipeline(pid)
        all_results[pid]    = {"splits": clean_splits, "leaks": leaks}

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "="*64)
    print("  STEP 1 COMPLETE — SUMMARY")
    print("="*64)
    print(f"\n  {'Pipeline':<12} {'Train':>8} {'Dev':>8} {'Test':>8}  Leaks")
    print(f"  {'-'*55}")
    for pid in args.pipeline:
        r = all_results[pid]
        s = r["splits"]
        l = r["leaks"]
        leak_str = ("NONE" if all(v == 0 for v in l.values())
                    else f"⚠ tr↔dev={l['train_dev']} tr↔tst={l['train_test']}")
        print(f"  {pid:<12} "
              f"{len(s['train']):>8,} {len(s['dev']):>8,} {len(s['test']):>8,}"
              f"  {leak_str}")
    print(f"\n  Figures   → {FIGURES_DIR}/pipeline_<ID>/")
    print(f"  Stats CSV → {OUTPUT_DIR}/pipeline_<ID>/corpus_statistics*.csv")
    print("="*64 + "\n")


if __name__ == "__main__":
    main()
