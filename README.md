# nanoGPT-Arabic

**A 134M-parameter Arabic GPT trained from scratch on Karpathy's nanoGPT, using AraGPT2's tokenizer and Arabic Wikipedia.**

---

## About

This project adapts [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) — a minimal, hackable GPT-2 implementation — to train an Arabic language model from scratch on a single RTX 4080 Laptop GPU.

**The training infrastructure (`train.py`, `model.py`, `sample.py`, `configurator.py`) is Karpathy's. All credit for the core nanoGPT code goes to him.** What's added here is:

- An Arabic data pipeline (`data/arabic/`) that downloads Arabic Wikipedia and tokenizes it with the AraGPT2 BPE tokenizer
- An Arabic-specific training config (`config/train_arabic_aragpt2.py`)
- A custom inference script (`sample_arabic.py`) that uses the AraGPT2 tokenizer instead of tiktoken

The original repo trains GPT-2 on English (OpenWebText / Shakespeare). This fork trains a comparable model on Arabic.

---

## What it does

Given a prompt in Arabic, generates a continuation:

```
Prompt:  في عام
Output:  في عام 1998، حصل على بكالوريوس التربية من جامعة كاليفورنيا،
         وحصل على بكالوريوس الآداب في العلوم من جامعة كاليفورنيا...
```

The model produces grammatically valid Modern Standard Arabic with correct verb conjugation, gender agreement, and *iḍāfa* (possessive) constructions. It captures Wikipedia-style patterns like biographies, year ranges, and proper nouns.

Limitations: factual hallucination (model is too small to know real history reliably) and repetition loops on rare topics. These are expected for the model size (134M) and training duration (~80 minutes).

---

## Architecture & training

| Component | Value |
|---|---|
| Architecture | GPT-2 small (12 layers, 12 heads, 768 embed dim) |
| Parameters | 134.11M |
| Tokenizer | AraGPT2 BPE (`aubmindlab/aragpt2-base`) |
| Vocab size | 64,000 |
| Context length | 512 tokens |
| Dataset | Arabic Wikipedia (`wikimedia/wikipedia 20231101.ar`) |
| Dataset size | 2.95 GB raw text → 467.8M tokens |
| Train/val split | 421M / 46.8M tokens |
| Compression | 6.31 bytes/token |
| Effective batch size | 132 sequences |
| Training duration | 2,000 iterations (~80 min on RTX 4080 Laptop) |
| Final loss | train 3.39 / val 3.73 |
| dtype | bfloat16 (Ada Lovelace native support) |

---

## Hardware

- GPU: NVIDIA GeForce RTX 4080 Laptop (12 GB VRAM, compute capability 8.9)
- OS: Windows 11
- Python: 3.10
- PyTorch: 2.5.1 + CUDA 12.1
- Conda environment

---

## Setup

```powershell
# 1. Clone Karpathy's nanoGPT and add this fork's files
git clone https://github.com/karpathy/nanoGPT.git
cd nanoGPT

# 2. Create environment
conda create -n nanogpt python=3.10 -y
conda activate nanogpt

# 3. Install dependencies
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install numpy -y
pip install transformers datasets tiktoken wandb tqdm

# 4. Pin sympy version (PyTorch 2.5.1 requires this)
pip install "sympy==1.13.1"
```

Verify GPU is visible:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Usage

The pipeline runs in three steps. Each is independent — you can re-run any of them.

### 1. Download Arabic Wikipedia

```powershell
python data\arabic\get_data.py
```

Downloads ~3 GB of Arabic Wikipedia text into `data/arabic/input.txt`. Takes ~10 minutes once HuggingFace caches the parquet shards.

### 2. Tokenize with AraGPT2 BPE

```powershell
python data\arabic\prepare.py
```

Loads the AraGPT2 tokenizer (downloaded once on first run, ~8 MB), reads `input.txt` in 1 MB chunks, encodes to token IDs, and writes:

- `train.bin` — 421M training tokens (uint16)
- `val.bin` — 46.8M validation tokens
- `meta.pkl` — vocab size + tokenizer ID, read automatically by `train.py`

Takes ~1.5 hours for 3 GB. Memory peak ~6 GB RAM.

### 3. Train

```powershell
python train.py config\train_arabic_aragpt2.py
```

Trains a 134M GPT from random initialization. Reads `meta.pkl` automatically to set vocab size. Saves the best checkpoint to `out-arabic-aragpt2/ckpt.pt` whenever val loss improves.

Takes ~80 minutes for 2000 iters on RTX 4080 Laptop at ~2.4s/iter (MFU ~7.6%).

### 4. Sample

```powershell
python sample_arabic.py
```

Loads the checkpoint and generates 3 continuations of `"في عام"` ("In the year"). Edit the `start` variable inside `sample_arabic.py` to change the prompt:

```python
start = "بسم الله"                      # Religious
start = "المملكة العربية السعودية"     # Saudi Arabia
start = "في القرن الحادي والعشرين"     # In the 21st century
```

---

## File map

What's from Karpathy's nanoGPT (unchanged):
- `train.py` — training loop, DDP, AMP, checkpointing
- `model.py` — GPT class, ~300 lines, `CausalSelfAttention` + `MLP` blocks
- `sample.py` — original GPT-2 sampling (uses tiktoken; not used in this project)
- `configurator.py` — the "poor man's argparse" that loads config files via `exec()`

What's added for Arabic:
- `data/arabic/get_data.py` — downloads Arabic Wikipedia
- `data/arabic/prepare.py` — chunked tokenization with AraGPT2 BPE
- `config/train_arabic_aragpt2.py` — from-scratch training config for Arabic
- `sample_arabic.py` — inference using the AraGPT2 tokenizer

---

## Sample outputs

After 2000 training iterations (final train loss 3.39):

**Prompt: `في عام` ("In the year")**

> في عام 1998، حصل على بكالوريوس التربية من جامعة كاليفورنيا، وحصل على بكالوريوس الآداب في العلوم من جامعة كاليفورنيا، كما حصل على درجة البكالوريوس في الدراسات العليا في العلوم من جامعة ييل...

> في عام 1837، كان هناك القليل من المباني السكنية الصغيرة. هناك عدد كبير من المباني التي كانت متاحة فقط في مدينة نيويورك...

The Arabic morphology, possessive constructions, and Wikipedia-style formatting are all correct. The model invents specific facts (it's a 134M model, not a knowledge base) but the language is fluent.

---

## Why this works (the technical decisions)

**Why train from scratch instead of fine-tuning OpenAI GPT-2?**
GPT-2's BPE tokenizer is English-only. When fed Arabic, it falls back to UTF-8 byte tokens — every Arabic character becomes ~2 tokens of garbage that the model has barely seen. A 512-token context holds only ~80 Arabic characters that way. We tried it as a baseline; loss dropped from 2.31 → 2.04 in 100 iters, but generations were mostly malformed.

**Why AraGPT2's tokenizer specifically?**
AUB MIND Lab trained AraGPT2 on ~77 GB of Arabic text and built a 64K-vocab BPE tokenizer in the process. It captures Arabic morphemes (`ال`, `ون`, common roots) as single tokens. On our corpus it achieves **6.31 bytes/token** vs **1.7** for byte-level GPT-2 — roughly 3.7× more efficient, meaning 3.7× more Arabic semantic content per training step.

**Why 134M params on 467M tokens?**
The Chinchilla scaling law suggests roughly 20 tokens per parameter as compute-optimal. 467M ÷ 134M ≈ 3.5 tokens/param — we're undertrained, not overparameterized. With more time (10–20K iters) the same model would continue improving. 2000 iters was a time-bounded delivery choice.

**Why bfloat16?**
RTX 4080 (Ada Lovelace, compute capability 8.9) supports bf16 natively. Faster than fp32, more numerically stable than fp16 (no loss-scaling needed). Free win.

**Why no `torch.compile`?**
Doesn't work on Windows — Triton (the underlying kernel compiler) has no Windows support. Cost: ~30% slower per iter. Acceptable.

---

## Acknowledgments

- **Andrej Karpathy** — for nanoGPT, the cleanest GPT-2 implementation in existence. This project is a thin Arabic-specific wrapper around his code.
- **AUB MIND Lab** — for AraGPT2 and its tokenizer ([aubmindlab/aragpt2-base](https://huggingface.co/aubmindlab/aragpt2-base)).
- **Wikimedia Foundation** — for the Arabic Wikipedia dump used as training data.
- **Anthropic Claude** — used as a debugging pair throughout the build.

---

## License

Same as nanoGPT (MIT). Trained model weights, if shared, are derivatives of Wikipedia (CC BY-SA 4.0) and AraGPT2 (which is permissively licensed for tokenizer use).
