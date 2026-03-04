"""
Fine-tune ClinicalT5 on local discharge-note style with LoRA, then export a
merged encoder for embedding generation (used by 02_embed.py).

Output:
  models/clinical_t5_finetuned/adapter/  (LoRA adapter)
  models/clinical_t5_finetuned/encoder/  (merged T5EncoderModel)
  models/clinical_t5_finetuned/info.json
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    T5EncoderModel,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

try:
    from .config import FEATURES_CSV, MIMIC_BHC_DIR, MIMIC_NOTE_DIR, MODELS_DIR, RANDOM_STATE
except ImportError:
    from config import FEATURES_CSV, MIMIC_BHC_DIR, MIMIC_NOTE_DIR, MODELS_DIR, RANDOM_STATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL = "luqh/ClinicalT5-base"
MAX_TRAIN_SAMPLES = 60_000
MAX_TEXT_CHARS = 3000
MIN_TEXT_LEN = 80
TRAIN_EPOCHS = 2
TRAIN_BS = 2
GRAD_ACC = 8
LR = 2e-4
MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 192
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
VAL_FRAC = 0.05

FINETUNE_DIR = os.path.join(MODELS_DIR, "clinical_t5_finetuned")
ADAPTER_DIR = os.path.join(FINETUNE_DIR, "adapter")
ENCODER_DIR = os.path.join(FINETUNE_DIR, "encoder")
INFO_JSON = os.path.join(FINETUNE_DIR, "info.json")

SECTIONS = [
    "brief hospital course", "hospital course", "discharge diagnosis",
    "discharge condition", "assessment and plan", "assessment/plan",
    "history of present illness", "past medical history",
]


def preprocess_note(text: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    if not isinstance(text, str) or len(text.strip()) < MIN_TEXT_LEN:
        return ""
    text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)
    text = text.replace("\r", "\n")
    text_lower = text.lower()
    out: List[str] = []
    for sec in SECTIONS:
        pat = re.compile(
            rf"{re.escape(sec)}\s*[:\-]?\s*(.*?)(?=\n[A-Z][A-Z /]{{2,40}}\s*[:\-]|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        m = pat.search(text_lower)
        if not m:
            continue
        s, e = m.span(1)
        block = re.sub(r"\s+", " ", text[s:e]).strip()
        if block:
            out.append(block[:1200])
    if out:
        note = " ".join(out)
    else:
        note = re.sub(r"\s+", " ", text).strip()
    return note[:max_chars]


def _corrupt(text: str) -> str:
    words = text.split()
    if len(words) < 20:
        return text
    kept = []
    for w in words:
        if random.random() > 0.15:
            kept.append(w)
    if len(kept) < 10:
        kept = words[:]
    return " ".join(kept)


def build_file_index(dirs: List[str]) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                idx.setdefault(f, os.path.join(root, f))
    return idx


def load_training_texts() -> List[str]:
    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(f"Missing {FEATURES_CSV}. Run 01_extract.py first.")
    cohort = set(pd.read_csv(FEATURES_CSV, usecols=["hadm_id"])["hadm_id"].astype(int))
    idx = build_file_index([MIMIC_NOTE_DIR, MIMIC_BHC_DIR])

    notes = pd.DataFrame(columns=["hadm_id", "text"])
    for fn in ["discharge.csv.gz", "discharge.csv"]:
        p = idx.get(fn)
        if not p:
            continue
        logger.info("Loading notes from %s", p)
        df = pd.read_csv(p, usecols=["hadm_id", "text"], nrows=1_500_000, low_memory=True)
        df = df[df["hadm_id"].isin(cohort)].dropna(subset=["text"])
        df["text"] = df["text"].map(preprocess_note)
        df = df[df["text"].str.len() >= MIN_TEXT_LEN]
        notes = df.groupby("hadm_id")["text"].apply(" ".join).reset_index()
        break

    if notes.empty:
        raise RuntimeError("No training notes found.")

    texts = notes["text"].tolist()
    if len(texts) > MAX_TRAIN_SAMPLES:
        rng = np.random.RandomState(RANDOM_STATE)
        sel = rng.choice(len(texts), size=MAX_TRAIN_SAMPLES, replace=False)
        texts = [texts[i] for i in sel]
    logger.info("Prepared %d notes for LoRA fine-tuning.", len(texts))
    return texts


@dataclass
class Tokenized:
    tokenizer: AutoTokenizer

    def __call__(self, batch):
        src = [f"summarize: {_corrupt(t)}" for t in batch["text"]]
        tgt = [t for t in batch["text"]]
        model_inputs = self.tokenizer(
            src, max_length=MAX_INPUT_LEN, truncation=True, padding=False
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt, max_length=MAX_TARGET_LEN, truncation=True, padding=False
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def export_merged_encoder(merged_seq2seq: T5ForConditionalGeneration, tokenizer: AutoTokenizer) -> None:
    encoder = T5EncoderModel.from_pretrained(BASE_MODEL, from_flax=True)
    full_state = merged_seq2seq.state_dict()
    enc_state = {k.replace("encoder.", "", 1): v for k, v in full_state.items() if k.startswith("encoder.")}
    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    if missing:
        logger.warning("Encoder export missing keys: %d", len(missing))
    if unexpected:
        logger.warning("Encoder export unexpected keys: %d", len(unexpected))
    os.makedirs(ENCODER_DIR, exist_ok=True)
    encoder.save_pretrained(ENCODER_DIR)
    tokenizer.save_pretrained(ENCODER_DIR)
    logger.info("Merged encoder saved to %s", ENCODER_DIR)


def main():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    os.makedirs(FINETUNE_DIR, exist_ok=True)

    texts = load_training_texts()
    ds = Dataset.from_dict({"text": texts})
    split = ds.train_test_split(test_size=VAL_FRAC, seed=RANDOM_STATE)
    train_ds = split["train"]
    val_ds = split["test"]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = T5ForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        from_flax=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    peft_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q", "k", "v", "o"],
        bias="none",
    )
    model = get_peft_model(base_model, peft_cfg)
    model.print_trainable_parameters()

    tok = Tokenized(tokenizer=tokenizer)
    train_tok = train_ds.map(tok, batched=True, remove_columns=["text"])
    val_tok = val_ds.map(tok, batched=True, remove_columns=["text"])

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    args = TrainingArguments(
        output_dir=os.path.join(FINETUNE_DIR, "checkpoints"),
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=TRAIN_BS,
        per_device_eval_batch_size=TRAIN_BS,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to=[],
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()

    os.makedirs(ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    logger.info("LoRA adapter saved to %s", ADAPTER_DIR)

    merged = model.merge_and_unload()
    export_merged_encoder(merged, tokenizer)

    with open(INFO_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model": BASE_MODEL,
                "adapter_dir": ADAPTER_DIR,
                "encoder_dir": ENCODER_DIR,
                "epochs": TRAIN_EPOCHS,
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
            },
            f,
            indent=2,
        )
    logger.info("Fine-tune metadata saved to %s", INFO_JSON)


if __name__ == "__main__":
    main()
