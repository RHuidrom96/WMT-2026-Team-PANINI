# WMT-2026-Team-PANINI

# Manipuri MT — IndicTrans2 1B Fine-tuning (BT+FT+Iter+SubwordReg)

Low-resource machine translation between English and Manipuri, built on **IndicTrans2 1B** with **LoRA** fine-tuning via PEFT.

This repo contains **BT + FT + Iterative Pseudo-labelling + Subword Regularisation** — the final augmentation experiment in the series, combining back-translation, forward translation, and iterative pseudo-labelled data with training-time subword regularisation on top of a fine-tuned base adapter.

## Overview

- **Base model:** IndicTrans2 1B (encoder-decoder seq2seq)
- **Adapter method:** rsLoRA (PEFT)
- **Preprocessing:** `IndicTransToolkit.processor.IndicProcessor`

## Files

| File | Purpose |
|---|---|
| `config.py` | Central configuration (paths, hyperparameters) |
| `step5e_subword_reg.py` | BT+FT+Iter+SubwordReg experiment runner |
| `aug_techniques.py` | BT / FT / iterative pseudo-labelling generators |
| `aug_utils.py` | Shared helper utilities |
| `aug_finetune_eval.py` | Fine-tune + evaluation runner |
| `checkpoints` | https://drive.google.com/drive/folders/1PYGYSo-m9HrrJ5DNGjUb0evZtM7qH3_4?usp=sharing |

## Requirements

```bash
pip install requirements.txt
pip install torch transformers datasets peft sacrebleu unbabel-comet pandas numpy openpyxl
pip install git+https://github.com/VarunGumma/IndicTransToolkit.git
```

## Usage

```bash

python step5e_subword_reg.py --pipeline A or B or C or D
```

## Output

- test_data:  `outputs/pipeline_<ID>/test.xlsx`  
- Adapter: `checkpoints/pipeline_<ID>/exp5_subword_reg/`
- Predictions: `outputs/pipeline_<ID>/preds_exp5_subword_reg.csv`
- Scores: `outputs/all_scores.json` (key: `exp5_<pipeline>`)

---
