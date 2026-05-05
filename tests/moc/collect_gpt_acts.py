"""
collect_gpt_acts.py
====================
Run GPT-2 (small, 117 M) over a WikiText-103 sample and save the
last-layer residual-stream activations together with POS-tag labels.

Outputs (saved to tests/moc/gpt_acts_output/):
  acts/
    acts_train.npy   – (N_train, 768) float32 residual-stream vectors
    acts_test.npy    – (N_test,  768) float32
    pos_train.npy    – (N_train,)     int16   POS label indices
    pos_test.npy     – (N_test,)      int16
    token_train.npy  – (N_train,)     int32   GPT-2 token ids
    token_test.npy   – (N_test,)      int32
    pos_vocab.json   – {pos_tag: index} mapping

Configuration:
  MAX_TRAIN_TOKENS  – how many token-position activations to collect for train
  MAX_TEST_TOKENS   – same for test
  Adjust below to trade off disk space vs. coverage.
"""

import os, json, re
import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer
import spacy
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────

HERE      = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(HERE, "gpt_acts_output")
ACT_DIR   = os.path.join(OUT_DIR, "acts")
os.makedirs(ACT_DIR, exist_ok=True)

MODEL_NAME       = "gpt2"
LAYER            = -1          # last transformer block hidden state
MAX_TRAIN_TOKENS = 200_000     # token-position pairs to collect (train)
MAX_TEST_TOKENS  =  40_000     # (test)
BATCH_CHARS      = 2_048       # characters per text chunk fed to GPT-2
MAX_SEQ_LEN      = 256         # tokens per forward pass (truncated)
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
SEED             = 42

# ── Load spaCy for POS tagging ────────────────────────────────────────────────

print("Loading spaCy en_core_web_sm …")
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
except OSError:
    raise RuntimeError(
        "spaCy model not found. Run: python -m spacy download en_core_web_sm"
    )

# Universal POS tag set (17 tags)
UPOS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
    "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
    "SCONJ", "SYM", "VERB", "X",
]
POS_TO_IDX = {t: i for i, t in enumerate(UPOS_TAGS)}

# ── Load GPT-2 ────────────────────────────────────────────────────────────────

print(f"Loading {MODEL_NAME} …")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2Model.from_pretrained(MODEL_NAME, output_hidden_states=True)
model.eval().to(DEVICE)
D_MODEL = model.config.hidden_size  # 768

print(f"  d_model={D_MODEL}, device={DEVICE}")

# ── Load WikiText-103 ─────────────────────────────────────────────────────────

print("Loading WikiText-103 …")
ds_train = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
ds_test  = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

def iter_nonempty(ds):
    for row in ds:
        text = row["text"].strip()
        if len(text) > 20:
            yield text

# ── Helpers ───────────────────────────────────────────────────────────────────

def gpt2_token_to_spacy_char_offsets(gpt2_tokens, text):
    """
    Return a list of (char_start, char_end) for each GPT-2 token in `text`.
    GPT-2 uses byte-level BPE; the tokenizer's convert_ids_to_tokens gives
    pieces with Ġ for leading space.  We map them back to character offsets
    by a greedy left-to-right scan.
    """
    pieces = tokenizer.convert_ids_to_tokens(gpt2_tokens)
    offsets = []
    cursor = 0
    for piece in pieces:
        raw = piece.replace("Ġ", " ").replace("Ċ", "\n")
        # skip whitespace that the piece absorbed
        idx = text.find(raw, cursor)
        if idx == -1:
            # fallback: advance cursor by 1 to avoid infinite loop
            offsets.append((cursor, cursor))
            continue
        offsets.append((idx, idx + len(raw)))
        cursor = idx + len(raw)
    return offsets


def collect_activations(ds_iter, max_tokens, label=""):
    """Collect up to max_tokens (activation, pos_label, token_id) triples."""
    all_acts  = []
    all_pos   = []
    all_toks  = []
    total     = 0

    for text in ds_iter:
        if total >= max_tokens:
            break

        # ── 1. spaCy POS for every character ──────────────────────────────
        doc = nlp(text)
        # build char→POS map
        char_pos = {}
        for token in doc:
            for ci in range(token.idx, token.idx + len(token.text)):
                char_pos[ci] = token.pos_

        # ── 2. GPT-2 tokenise ─────────────────────────────────────────────
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding=False,
        )
        input_ids = enc["input_ids"].to(DEVICE)   # (1, T)
        T = input_ids.shape[1]

        # ── 3. Forward pass ───────────────────────────────────────────────
        with torch.no_grad():
            out = model(**enc.to(DEVICE))
        # hidden_states: tuple of (n_layers+1) tensors, each (1, T, 768)
        hidden = out.hidden_states[LAYER]  # (1, T, 768)
        acts = hidden[0].float().cpu().numpy()   # (T, 768)

        # ── 4. Align GPT-2 tokens → spaCy POS ────────────────────────────
        token_ids_np = input_ids[0].cpu().numpy()
        try:
            offsets = gpt2_token_to_spacy_char_offsets(token_ids_np.tolist(), text)
        except Exception:
            continue

        for t_idx in range(T):
            c_start, _ = offsets[t_idx]
            pos_tag = char_pos.get(c_start, "X")
            pos_idx = POS_TO_IDX.get(pos_tag, POS_TO_IDX["X"])
            all_acts.append(acts[t_idx])
            all_pos.append(pos_idx)
            all_toks.append(int(token_ids_np[t_idx]))
            total += 1
            if total >= max_tokens:
                break

        if total % 10_000 < T:
            print(f"  {label}: {total}/{max_tokens} tokens collected")

    acts_arr  = np.array(all_acts,  dtype=np.float32)
    pos_arr   = np.array(all_pos,   dtype=np.int16)
    toks_arr  = np.array(all_toks,  dtype=np.int32)
    return acts_arr, pos_arr, toks_arr


# ── Collect ───────────────────────────────────────────────────────────────────

print(f"\nCollecting train activations (up to {MAX_TRAIN_TOKENS:,} tokens) …")
acts_train, pos_train, tok_train = collect_activations(
    iter_nonempty(ds_train), MAX_TRAIN_TOKENS, label="train")

print(f"\nCollecting test activations (up to {MAX_TEST_TOKENS:,} tokens) …")
acts_test, pos_test, tok_test = collect_activations(
    iter_nonempty(ds_test), MAX_TEST_TOKENS, label="test")

# ── Shuffle train ─────────────────────────────────────────────────────────────

rng = np.random.default_rng(SEED)
perm = rng.permutation(len(acts_train))
acts_train = acts_train[perm]
pos_train  = pos_train[perm]
tok_train  = tok_train[perm]

# ── Save ──────────────────────────────────────────────────────────────────────

print("\nSaving …")
np.save(os.path.join(ACT_DIR, "acts_train.npy"), acts_train)
np.save(os.path.join(ACT_DIR, "acts_test.npy"),  acts_test)
np.save(os.path.join(ACT_DIR, "pos_train.npy"),  pos_train)
np.save(os.path.join(ACT_DIR, "pos_test.npy"),   pos_test)
np.save(os.path.join(ACT_DIR, "token_train.npy"), tok_train)
np.save(os.path.join(ACT_DIR, "token_test.npy"),  tok_test)
with open(os.path.join(ACT_DIR, "pos_vocab.json"), "w") as f:
    json.dump(POS_TO_IDX, f, indent=2)

print(f"  acts_train : {acts_train.shape}  dtype={acts_train.dtype}")
print(f"  acts_test  : {acts_test.shape}   dtype={acts_test.dtype}")
print(f"  pos_train  : {pos_train.shape}")
print(f"  pos_test   : {pos_test.shape}")
print(f"Saved to {ACT_DIR}")
