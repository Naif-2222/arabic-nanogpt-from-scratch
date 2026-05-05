"""Tokenize input.txt with AraGPT2 BPE tokenizer, chunked for memory safety."""
import os
import pickle
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

DATA_DIR  = os.path.dirname(os.path.abspath(__file__))
INPUT     = os.path.join(DATA_DIR, "input.txt")
MODEL_ID  = "aubmindlab/aragpt2-base"
CHUNK_SIZE = 1_000_000  # chars per chunk; ~3MB; safe RAM-wise

print(f"Loading AraGPT2 tokenizer ({MODEL_ID})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
vocab_size = tokenizer.vocab_size
print(f"Vocab size: {vocab_size}")

# get file size for progress bar
file_size = os.path.getsize(INPUT)
print(f"Input file: {file_size/1024/1024:.1f} MB")

# pass 1: tokenize whole file, write all ids first to temp lists
all_ids = []
with open(INPUT, "r", encoding="utf-8") as f:
    pbar = tqdm(total=file_size, unit="B", unit_scale=True, desc="tokenizing")
    while True:
        chunk = f.read(CHUNK_SIZE)
        if not chunk:
            break
        ids = tokenizer.encode(chunk)
        all_ids.append(np.array(ids, dtype=np.uint32))
        pbar.update(len(chunk.encode("utf-8")))
    pbar.close()

all_ids = np.concatenate(all_ids)
print(f"Total tokens: {len(all_ids):,}")

# 90/10 split
split_idx = int(len(all_ids) * 0.9)
train_ids = all_ids[:split_idx]
val_ids   = all_ids[split_idx:]
print(f"train: {len(train_ids):,} tokens")
print(f"val:   {len(val_ids):,} tokens")
print(f"compression: {file_size / len(all_ids):.2f} bytes/token")

# uint16 if vocab < 65536, else uint32
dtype = np.uint16 if vocab_size < 65536 else np.uint32
train_ids.astype(dtype).tofile(os.path.join(DATA_DIR, "train.bin"))
val_ids.astype(dtype).tofile(os.path.join(DATA_DIR, "val.bin"))

meta = {"vocab_size": vocab_size, "tokenizer_id": MODEL_ID}
with open(os.path.join(DATA_DIR, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)
print(f"Wrote train.bin, val.bin, meta.pkl (vocab_size={vocab_size})")