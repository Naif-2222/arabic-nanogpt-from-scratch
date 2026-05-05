"""Download a chunk of Arabic Wikipedia as plain text into input.txt"""
import os
from datasets import load_dataset
from tqdm import tqdm

TARGET_MB = 3000  # ~20MB is plenty for a fine-tune demo
out_path = os.path.join(os.path.dirname(__file__), "input.txt")

ds = load_dataset("wikimedia/wikipedia", "20231101.ar", split="train", streaming=True)

target_bytes = TARGET_MB * 1024 * 1024
written = 0
with open(out_path, "w", encoding="utf-8") as f:
    for row in tqdm(ds, desc="downloading"):
        text = row["text"].strip()
        if not text:
            continue
        f.write(text + "\n\n")
        written += len(text.encode("utf-8"))
        if written >= target_bytes:
            break

print(f"Wrote {written/1024/1024:.1f} MB to {out_path}")