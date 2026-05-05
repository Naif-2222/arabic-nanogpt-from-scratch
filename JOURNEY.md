# JOURNEY.md — Building nanoGPT-Arabic, the engineering story

This is the honest account of what it took to get this working: the dead-ends, the mistakes, and the fixes. Written because the polished `README.md` makes it look easier than it was, and because the debugging is where the real learning happened.

This project sits on top of [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT). His training infrastructure is excellent. But adapting it for Arabic, on Windows, on a single laptop, surfaced a series of problems that aren't in the original README.

---

## The starting point

Goal: take Karpathy's nanoGPT, train it to generate Arabic.

Hardware: Windows 11, RTX 4080 Laptop (12 GB VRAM), 32 GB RAM.

Initial assumption: "It's nanoGPT, the README has commands, this should be a Sunday afternoon."

Actual time to working Arabic generation: ~2 days of elapsed time, ~6 hours of active debugging, plus ~80 minutes of training.

---

## Struggle 1 — `<3` is not a version constraint

The nanoGPT README lists dependencies like:

```
pytorch <3
numpy <3
```

I read `<3` as "less than 3" — a version constraint. It's actually a heart emoji. Karpathy is saying he loves these libraries.

**Lesson:** read READMEs like prose, not specifications. The `<3` is decorative, not a version spec.

---

## Struggle 2 — Conda vs pip on Windows

First tried `pip install torch`. Got the CPU-only wheel by default. `torch.cuda.is_available()` returned `False`. Reinstalled with the explicit CUDA index URL. Worked.

Then ran into a sympy version conflict:

```
torch 2.5.1 requires sympy==1.13.1, but you have sympy 1.14.0
```

PyTorch's wheel pins `sympy==1.13.1` exactly, but conda installed 1.14.0 transitively. Fixed with `pip install "sympy==1.13.1"`.

**Lesson:** on Windows, prefer conda for PyTorch + CUDA, but be ready to pin one or two transitive deps with pip after.

---

## Struggle 3 — Running config files as if they were scripts

Tried to start training with:

```powershell
python config/finetune_shakespeare.py
```

Nothing happened. No error. Just silence. Spent several minutes confused.

**The cause:** config files in nanoGPT are NOT scripts. They are Python files containing variable assignments that get loaded by `train.py` via `exec()` (in `configurator.py`). Running them directly executes assignments and exits.

The correct command is:

```powershell
python train.py config/finetune_shakespeare.py
```

`train.py` is the script. The config is an argument.

**Lesson:** in nanoGPT, three scripts only ever run: `data/<dataset>/prepare.py`, `train.py`, `sample.py`. Everything else is data or config.

---

## Struggle 4 — Accidentally downloading 54 GB of OpenWebText

Misreading the README, ran:

```powershell
python data/openwebtext/prepare.py
```

This downloads the **English** OpenWebText dataset (54 GB, 8M documents) for reproducing GPT-2 from scratch. It started downloading parquet files at ~85 KB/s. At that rate the download would take 20+ hours.

Realized this had nothing to do with the Arabic task. Killed it with Ctrl+C. Deleted the partial cache from `~/.cache/huggingface/`.

**Lesson:** read the directory names. `data/openwebtext/` is for English GPT-2 reproduction. `data/shakespeare/` and `data/shakespeare_char/` are for character-level demos. We needed `data/arabic/` — which didn't exist yet because we hadn't created it.

---

## Struggle 5 — HuggingFace streaming dataset disconnections

First Arabic data approach: stream Arabic Wikipedia via `load_dataset("wikimedia/wikipedia", "20231101.ar", split="train", streaming=True)`.

Hit repeated disconnections:

```
'[WinError 10054] An existing connection was forcibly closed by the remote host'
```

Streaming downloads don't resume cleanly when connections drop. Multiple retries gave inconsistent results — sometimes 300 MB, sometimes a traceback.

**Fix:** switched to non-streaming mode (`streaming=False`). This downloads the whole dataset to local cache first, then iterates from disk. Once cached, no network = no disconnects.

```python
ds = load_dataset("wikimedia/wikipedia", "20231101.ar", split="train")  # no streaming
```

This was much more reliable. The full Arabic Wikipedia (~1.22M articles) cached in ~10 minutes once HF was happy.

**Lesson:** for flaky connections or large datasets, streaming is fragile. Cache to disk, then iterate.

---

## Struggle 6 — Empty config file produced silent default training

Created `config/finetune_arabic.py` with the right content. Ran training. Got two errors at once:

```
Initializing a new model from scratch       ← should say "from gpt2"
defaulting to vocab_size of GPT-2 to 50304   ← should be from meta.pkl
FileNotFoundError: 'data\openwebtext\train.bin'   ← should be data/arabic/
compiling the model...                        ← compile=False was supposed to disable this
```

Verified the config file:

```powershell
type config\finetune_arabic.py
```

The output was empty. The file existed but had zero bytes. The "Save" in my editor hadn't taken effect, or the encoding had nuked the content.

`train.py` silently used all defaults: `init_from='scratch'`, `dataset='openwebtext'`, `compile=True`. None of which were what we wanted.

**Fix:** verified file content with `type` after every save. From then on, never trusted that "save" actually saved.

**Lesson:** when training behavior surprises you, the first thing to check is whether your config is actually being loaded. `train.py` defaults are silent and will train on the wrong thing if you let them.

---

## Struggle 7 — PowerShell BOM corruption

Created the config file using PowerShell's heredoc syntax:

```powershell
@'
import time
out_dir = 'out-arabic'
...
'@ | Out-File -Encoding utf8 config\finetune_arabic.py
```

`Out-File -Encoding utf8` writes a UTF-8 **byte-order mark** (BOM) at the start of the file — three invisible bytes (`EF BB BF`). Python sees this and crashes:

```
SyntaxError: invalid character '»' (U+00BB)
```

Tried `Set-Content -Encoding utf8NoBOM` — but that switch only exists in PowerShell 7+. On Windows PowerShell 5.1, no luck.

**Fix:** wrote the file using Python directly:

```powershell
python -c "open('config/finetune_arabic.py','w',encoding='utf-8',newline='\n').write('''...''')"
```

Python's default UTF-8 has no BOM. Verified with:

```powershell
python -c "print(repr(open('config/finetune_arabic.py','rb').read()[:10]))"
# b'import tim'   ← correct, no BOM
```

**Lesson:** Windows PowerShell + Python source files = always check for BOMs. The safe ways to write a `.py` file are: Python's own `open(..., 'w', encoding='utf-8')`, VS Code, or Notepad (Windows 11 only — older Notepad versions add BOMs).

---

## Struggle 8 — `torch.compile` and Triton on Windows

After fixing the config, the next error:

```
RuntimeError: Cannot find a working triton installation.
torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised: ...
```

`torch.compile` (PyTorch 2.x) uses Triton as its kernel compiler. **Triton has no Windows build.** On Windows, `compile=True` always fails.

**Fix:** set `compile = False` in every config. Cost: ~30% slower iter time. Acceptable.

**Lesson:** on Windows, `compile = False` is mandatory in nanoGPT. This is not optional. Bake it into every config file.

---

## Struggle 9 — Tokenizer choice and the byte-fallback trap

First Arabic training run used Karpathy's `data/shakespeare/prepare.py` pattern: `tiktoken.get_encoding("gpt2").encode_ordinary(...)` on Arabic text.

This **technically works** — tiktoken handles arbitrary UTF-8 via byte fallback. But each Arabic character becomes 2 tokens (because UTF-8 encoding of Arabic is 2 bytes/char), and those tokens are essentially raw bytes that GPT-2's embeddings have barely seen.

Result: 173M tokens for 300 MB of text (~1.7 bytes/token). Fine-tuning GPT-2 on this dropped loss from 2.31 → 2.04 in 100 iters, but generations were mostly malformed Arabic letter sequences with occasional real words.

**Fix:** switched to AraGPT2's BPE tokenizer (`aubmindlab/aragpt2-base`). Result: 467M tokens for 2.95 GB (~6.3 bytes/token), 64K Arabic-aware vocabulary. Words like `السلام` ("the peace") become 1–2 tokens instead of 12+ bytes.

This required abandoning fine-tuning of GPT-2 (different vocab, incompatible embeddings) and switching to from-scratch training. But the quality jump was night and day.

**Lesson:** for any non-English language, the tokenizer choice dominates everything else. Byte-level BPE on a non-English language is a tax on every training step. Use a language-appropriate tokenizer or train one yourself.

---

## Struggle 10 — VRAM spillover, the silent killer

First from-scratch run with 134M params, `batch_size=16`, `block_size=512`. Iter time: **8–13 seconds**. MFU: **1.5%**. At that rate, 20K iterations = 55 hours.

Diagnostic: ran `nvidia-smi` in a second terminal. VRAM showed 11.8/12 GB. Windows Task Manager → GPU showed "Shared GPU memory" being used (~3 GB).

**Diagnosis:** when VRAM is full, Windows transparently spills tensors to system RAM via PCIe. PCIe is ~20–30× slower than VRAM. Once you're over the line, every iteration crawls.

**Fix:** dropped `batch_size = 16` → `10`, raised `gradient_accumulation_steps = 8` → `16` (kept effective batch at 160 sequences). VRAM dropped to ~7 GB. Iter time dropped to **2.9 seconds**. MFU jumped to **7.6%** — a 5× speedup.

**Lesson:** bigger batch size is only better if you fit. The "optimal" batch size on a given GPU is the largest that does NOT spill. Find it empirically by watching nvidia-smi. Going one step over is catastrophically slower.

---

## Struggle 11 — LR schedule mismatch

Config had `max_iters = 5000` but `lr_decay_iters = 20000`. This means: stop training at 5000 steps, but the LR schedule is calibrated to slowly cool down over 20000 steps. Result: at iter 5000, LR is still ~85% of peak — the model never gets the cooldown phase that lets it settle.

Easy to miss because it doesn't error, just produces worse results.

**Fix:** `lr_decay_iters` should always equal `max_iters` for a single training run, unless you're explicitly resuming a longer schedule.

**Lesson:** when you change `max_iters`, change `lr_decay_iters` to match. Always. They are not independent.

---

## Struggle 12 — High dropout slowing pretraining

Tried `dropout = 0.2` on the from-scratch run. Loss became noisy — bouncing between 3.4 and 5.9 across consecutive 25-iter windows, even after the warmup phase.

For from-scratch pretraining (not fine-tuning), high dropout adds gradient noise without enough signal-to-noise to compensate. Karpathy's defaults are 0.0 for pretraining, 0.1 for finetuning.

**Fix:** dropped to `dropout = 0.1`. Loss curve smoothed out.

**Lesson:** dropout helps fine-tuning (where overfitting is real). For from-scratch training on a large corpus, low or zero dropout is usually better.

---

## Struggle 13 — Repetition loops in samples

Final samples included repetitions like:

> في عام 2017، تم ترشيح فيلم قصير من قبل شركة ديزني، مع فيلم قصير من قبل شركة ديزني...

The model gets stuck in high-probability loops. This is not a model failure per se — it's a **sampling** issue. Karpathy's `sample.py` uses temperature + top-k, which doesn't aggressively penalize repetition.

**Partial fix (didn't fully implement):** could add `repetition_penalty` or `no_repeat_ngram_size` to the generation loop. We left this as a known limitation — the goal was to demonstrate Arabic generation, not to ship a production sampler.

**Lesson:** repetition in undertrained small LLMs is mostly a sampling problem, fixable without retraining. But it requires modifying the generation loop, which nanoGPT keeps deliberately minimal.

---

## What worked, in retrospect

- **Karpathy's `train.py`** is genuinely production-quality. Once the config and data were right, training Just Worked. DDP-ready, AMP, gradient accumulation, cosine LR schedule, checkpointing — all solid.
- **AraGPT2 tokenizer** was the single biggest quality multiplier. The 3.7× compression vs byte-level is a real, measured improvement.
- **bfloat16 on Ada Lovelace** is free performance. No loss scaling needed.
- **HuggingFace `datasets` non-streaming mode** is much more reliable than streaming for one-time downloads.
- **`nvidia-smi` in a second window** during training is the most informative debugging tool. Watch GPU-Util %, VRAM, and power draw.

---

## Total time accounting

| Phase | Time |
|---|---|
| Environment setup (conda, PyTorch, deps) | 30 min |
| Confused config debugging (struggles 6–8) | 2 hours |
| Wrong-tokenizer first attempt + cleanup | 1 hour |
| Arabic Wikipedia download (3 GB) | 15 min |
| Tokenization with AraGPT2 BPE | 1.5 hours |
| Diagnosing VRAM spillover, finding correct batch size | 30 min |
| Failed runs with wrong configs (LR mismatch, dropout 0.2) | 1 hour |
| Final successful training | 80 min |
| **Total elapsed (with dead time)** | **~2 days** |
| **Total active work** | **~7 hours** |

---

## Credits

- **Andrej Karpathy** — wrote nanoGPT. This project is his code with a small Arabic adapter on top. If you find this useful, [star his repo](https://github.com/karpathy/nanoGPT).
- **AUB MIND Lab** — wrote AraGPT2. Their tokenizer is the unsung hero of this project.
- The PyTorch team — for bf16 support and `F.scaled_dot_product_attention`.

The struggles documented here are not criticism of any of the above. They are the friction of running real systems on real hardware. Every one of them taught something. The polished version of this project lives in `README.md`. This file lives here because the polished version hides too much.
