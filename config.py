"""
config.py — Central Configuration (Direction-Specific Design)
─────────────────────────────────────────────────────────────────────────────
"""

import os
import torch

# ─── Pipeline definitions ──────────────────────────────────────────────────────
# Each entry: (pipeline_id, src_lang_code, tgt_lang_code, base_model_key)
PIPELINE_CONFIGS = {
    "A": {
        "src_lang":   "eng_Latn",
        "tgt_lang":   "mni_Mtei",
        "direction":  "en2indic",   
        "label":      "eng→mni_Mtei",
        "primary":    True,         
    },
    "B": {
        "src_lang":   "mni_Mtei",
        "tgt_lang":   "eng_Latn",
        "direction":  "indic2en",
        "label":      "mni_Mtei→eng",
        "primary":    True,
    },
    "C": {
        "src_lang":   "eng_Latn",
        "tgt_lang":   "mni_Beng",
        "direction":  "en2indic",
        "label":      "eng→mni_Beng",
        "primary":    True,
    },
    "D": {
        "src_lang":   "mni_Beng",
        "tgt_lang":   "eng_Latn",
        "direction":  "indic2en",
        "label":      "mni_Beng→eng",
        "primary":    True,
    },
}

ALL_PIPELINES    = list(PIPELINE_CONFIGS.keys())
PRIMARY_PIPELINE = "A"

# ─── Base HF checkpoints (one per direction family, shared across pipelines) ──
BASE_MODELS = {
    "en2indic":  "ai4bharat/indictrans2-en-indic-1B",
    "indic2en":  "ai4bharat/indictrans2-indic-en-1B",
}


def get_base_model(pipeline_id: str) -> str:
    """Return the HF model name for the given pipeline."""
    return BASE_MODELS[PIPELINE_CONFIGS[pipeline_id]["direction"]]


# ─── Paths ────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = "./outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# Raw input files (produced externally, one per pipeline)
RAW_DATA_DIR = "./data"

def raw_path(pipeline_id: str, split: str) -> str:
    """e.g. ./data/pipeline_A_train.xlsx"""
    return os.path.join(RAW_DATA_DIR, f"pipeline_{pipeline_id}_{split}.xlsx")

# Cleaned split files (output of step1)
def split_path(pipeline_id: str, split: str) -> str:
    return os.path.join(OUTPUT_DIR, f"pipeline_{pipeline_id}", f"{split}.xlsx")

# ─── Adapter / checkpoint dirs (one per pipeline) ────────────────────────────
def adapter_dir(pipeline_id: str, exp: str = "r1") -> str:
    return os.path.join("./checkpoints", f"pipeline_{pipeline_id}", exp)

# ─── Augmentation cache dirs (one sub-dir per pipeline) ──────────────────────
def synth_cache_dir(pipeline_id: str) -> str:
    return os.path.join(OUTPUT_DIR, f"pipeline_{pipeline_id}", "synth_cache")

def synth_cache_path(pipeline_id: str, technique: str) -> str:
    """e.g. outputs/pipeline_A/synth_cache/bt.xlsx"""
    return os.path.join(synth_cache_dir(pipeline_id), f"{technique}.xlsx")

# ─── Augmented dataset files (one per experiment per pipeline) ───────────────
def aug_dataset_path(pipeline_id: str, exp: str) -> str:
    return os.path.join(OUTPUT_DIR, f"pipeline_{pipeline_id}", f"aug_{exp}.xlsx")

# ─── Prediction output files ─────────────────────────────────────────────────
def preds_path(pipeline_id: str, system: str) -> str:
    return os.path.join(OUTPUT_DIR, f"pipeline_{pipeline_id}", f"preds_{system}.csv")

# ─── Scores file (shared across all pipelines & systems) ─────────────────────
SCORES_FILE = os.path.join(OUTPUT_DIR, "all_scores.json")

# ─── Data hyperparameters ─────────────────────────────────────────────────────
SEED        = 16
MAX_SEQ_LEN = 256

# ─── DoRA / LoRA configuration ────────────────────────────────────────────────
USE_DORA     = False   # True = DoRA (decomposed magnitude), False = LoRA
USE_RSLORA   = True    # rank-stabilized LoRA scaling
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
LORA_BIAS    = "none"
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "out_proj",
    "fc1", "fc2",
]

# ─── Training hyperparameters — R1 (step3) ────────────────────────────────────
NUM_TRAIN_EPOCHS        = 5
PER_DEVICE_TRAIN_BATCH  = 8
PER_DEVICE_EVAL_BATCH   = 16
GRAD_ACCUM_STEPS        = 4
LEARNING_RATE           = 2e-4
LR_SCHEDULER            = "cosine"
WARMUP_RATIO            = 0.05
WEIGHT_DECAY            = 0.01
MAX_GRAD_NORM           = 1.0
LABEL_SMOOTHING         = 0.1
SAVE_TOTAL_LIMIT        = 2
LOGGING_STEPS           = 25
EVAL_STEPS              = 200
SAVE_STEPS              = 200
EARLY_STOPPING_PATIENCE = 3

# ─── Training hyperparameters — Augmentation experiments ──────────────────────
NUM_TRAIN_EPOCHS_AUG = 3
LEARNING_RATE_AUG    = 1e-4

# ─── Precision ────────────────────────────────────────────────────────────────
BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
FP16 = (not BF16) and torch.cuda.is_available()

# ─── Inference ────────────────────────────────────────────────────────────────
INFER_BATCH_SIZE     = 32
NUM_BEAMS            = 5
MAX_NEW_TOKENS       = 256
LENGTH_PENALTY       = 1.0
NO_REPEAT_NGRAM_SIZE = 0

# ─── Metric configuration ─────────────────────────────────────────────────────
COMET_MODEL = "Unbabel/wmt22-comet-da"

# sacrebleu tokenization:
def sacrebleu_tokenize(pipeline_id: str) -> str:
    return "13a"
# ─── Augmentation quality-filter thresholds ───────────────────────────────────
BT_LABSE_THRESHOLD  = 0.75
QA_LABSE_THRESHOLD  = 0.75
ST_CONF_THRESHOLD   = 0.78

# ─── Augmentation sampling ratios ────────────────────────────────────────────
BT_SAMPLE_RATIO     = 0.40
FT_AUG_SAMPLE_RATIO = 0.30
ITER_SAMPLE_RATIO   = 0.25
NOISE_SAMPLE_RATIO  = 0.30

# ─── Synonym-replacement params (FT-Aug / Noise, source side) ────────────────
AUG_SR_PROB    = 0.15
AUG_N_PER_SENT = 1

# ─── Noise injection params (Step 5d) ────────────────────────────────────────
NOISE_SWAP_PROB    = 0.10   # word-swap probability
NOISE_DELETE_PROB  = 0.05   # word-deletion probability
NOISE_INSERT_PROB  = 0.05   # random word insertion probability

# ─── Subword regularisation (Step 5e) ────────────────────────────────────────
# Applied at training time via custom collator; no new data files generated.
SUBWORD_REG_ALPHA = 0.1    # SentencePiece sampling alpha (if model uses SP)

# ─── Hallucination analysis thresholds ───────────────────────────────────────
HALL_LEN_RATIO_LOW  = 0.4
HALL_LEN_RATIO_HIGH = 2.5
HALL_COPY_RATIO     = 0.8

SYSTEM_COLORS = {
    "baseline": "#C44E52",
    "r1":       "#4C72B0",
    "exp1":     "#55A868",
    "exp2":     "#DD8452",
    "exp3":     "#8172B2",
    "exp4":     "#937860",
}
SYSTEM_LABELS = {
    "baseline": "Baseline (zero-shot IT2-1B)",
    "r1":       "R1 (DoRA fine-tuned)",
    "exp1":     "Exp-1 (+BT)",
    "exp2":     "Exp-2 (+BT+FT)",
    "exp3":     "Exp-3 (+BT+FT+Iter)",
    "exp4":     "Exp-4 (+BT+FT+Iter+Noise)"
}

# ─── Make required directories on import ─────────────────────────────────────
for _pid in ALL_PIPELINES:
    for _d in [
        os.path.join(OUTPUT_DIR, f"pipeline_{_pid}"),
        synth_cache_dir(_pid),
        FIGURES_DIR,
        adapter_dir(_pid, "r1"),
        adapter_dir(_pid, "exp1_bt"),
        adapter_dir(_pid, "exp2_bt_ft"),
        adapter_dir(_pid, "exp3_bt_ft_iter"),
        adapter_dir(_pid, "exp4_noise"),
        adapter_dir(_pid, "step5e_subword"), #step5e_subword_reg.py
    ]:
        os.makedirs(_d, exist_ok=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs("./checkpoints", exist_ok=True)
