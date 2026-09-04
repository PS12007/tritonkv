# Running this benchmark on your GPU — a complete guide

Thanks for doing this. You're helping test something that genuinely can't be
tested on one computer.

**What you're doing, in one paragraph.** This project is a fast GPU program for
running AI text generation with a compressed memory cache. The finding it makes
is that the compression only pays off *once the cache is too big to fit in the
GPU's fast on-chip memory* — and how big that on-chip memory is differs from
card to card. It has only ever been measured on one laptop. Running it on your
machine, with a different GPU, is the only way to find out whether the
conclusion is about GPUs in general or just about that one laptop.

**What it costs you:** about 20 minutes of setup, then ~15 minutes where you
shouldn't touch the laptop. Then you send back a single 4 MB file.

**What it does NOT do:** no admin rights, no driver changes, no system-wide
installs, no overclocking, no registry edits. Everything lives in one folder you
can delete afterwards.

---

## TL;DR for the impatient

```bat
git clone https://github.com/PS12007/tritonkv
cd tritonkv
python -m venv .venv
.venv\Scripts\pip install torch==2.12.0+cu130 --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m pytest test_correctness.py -q
.venv\Scripts\python benchmark.py --samples 50
```

Send back `results/benchmark.json`. Plug the laptop in and don't use it during
the last command.

Everything below is the same thing with explanations and troubleshooting.

---

## What you need

| | |
|---|---|
| **GPU** | An NVIDIA card, RTX 20-series (2018) or newer. AMD and Intel graphics will not work. |
| **Driver** | Reasonably recent — version 580 or newer. Check with `nvidia-smi`. |
| **Python** | 3.10 or newer. 3.12 is what this was built on. |
| **Disk** | ~4 GB free. |
| **Time** | ~20 min setup (mostly downloading), ~15 min running. |
| **Power** | **Laptop must be plugged in.** See Step 4. |

You do **not** need admin rights, and you do **not** need to install CUDA
separately — the PyTorch package brings everything it needs.

---

## Step 0 — the 10-second check (do this before installing anything)

Open a terminal (Windows: press Start, type `cmd`, Enter) and run:

```bat
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```

You should get something like:

```
name, driver_version, memory.total [MiB]
NVIDIA GeForce RTX 4070 Laptop GPU, 581.29, 8188 MiB
```

**Send that line to Priyansh before you do anything else.** It takes ten seconds
and tells him whether your card is a useful test — it might save you the whole
install.

**If `nvidia-smi` is "not recognised":** your NVIDIA driver isn't installed, or
isn't on your PATH. Try the full path:
`"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"`. If that doesn't
exist either, install/update your GeForce driver first.

---

## Step 1 — get the code

**With git:**

```bat
git clone https://github.com/PS12007/tritonkv
cd tritonkv
```

**Without git:** go to <https://github.com/PS12007/tritonkv>, click the green
**Code** button → **Download ZIP**, extract it somewhere, then open a terminal in
that folder (in File Explorer, type `cmd` in the address bar and press Enter).

Put it somewhere with a simple path — `C:\tritonkv` is ideal. Avoid OneDrive
folders and paths with spaces or accented characters; they occasionally confuse
the GPU compiler.

---

## Step 2 — install the dependencies

This creates a private Python environment inside the project folder. Nothing is
installed system-wide.

```bat
python -m venv .venv
.venv\Scripts\pip install torch==2.12.0+cu130 --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\pip install -r requirements.txt
```

**This downloads about 3 GB** and takes 10–30 minutes depending on your
connection. It's mostly PyTorch and the CUDA libraries. It's normal for the first
command to sit at "Downloading torch..." for a long time.

**On Linux:** identical, except the paths use forward slashes
(`.venv/bin/pip`), and you need to edit `requirements.txt` first — change the
line `triton-windows==3.8.0.post28` to `triton==3.8.0`. There's a comment in the
file saying exactly this.

### Quick sanity check

Before going further, confirm PyTorch can actually see your GPU:

```bat
.venv\Scripts\python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected:

```
True
NVIDIA GeForce RTX 4070 Laptop GPU
```

If it prints `False`, stop here and read **Troubleshooting → PyTorch can't see
the GPU** below.

---

## Step 3 — check it actually works on your card (go/no-go)

```bat
.venv\Scripts\python -m pytest test_correctness.py -q
```

**Expected: `146 passed`.**

This takes **2–5 minutes the first time**. It's slow initially because the GPU
code is compiled on the fly for your specific card — that compilation is cached,
so later runs are much quicker. A long pause with no output is normal.

This step is genuinely important, not a formality: it verifies the kernel
produces numerically correct results *on your architecture*, which has never been
tested. If something is wrong with the code on your GPU, this is what catches it.

**If it fails:** stop, copy the error, send it to Priyansh. Don't run the
benchmark — the results would be meaningless. A failure here is almost always an
install problem, not a problem with your hardware.

---

## Step 4 — prepare your laptop (this part matters)

The benchmark measures *time*, very precisely. Anything else competing for the
GPU corrupts the measurement. Two minutes of prep:

- ✅ **Plug the laptop into mains power.** This is the big one. On battery the
  GPU deliberately throttles itself and the numbers are worthless.
- ✅ **Set Windows power mode to "Best performance"** (Settings → System → Power
  & battery → Power mode).
- ✅ **Close games, browsers with lots of tabs, Discord, OBS, video calls,
  anything using the GPU.** A browser playing video is enough to skew it.
- ✅ **Let the laptop cool down** if you've just been gaming. Starting hot
  changes the result.
- ✅ **Put it on a hard surface**, not a bed or cushion — it needs airflow for
  15 minutes at full load.
- ❌ **Don't use the laptop while it runs.** Go make a coffee.

**Don't worry about damaging anything.** This is a benchmark, not a stress test.
The GPU runs at full load for about 13 minutes — exactly like playing a game for
13 minutes. Fans will get loud and it'll get warm. That's normal and the laptop's
thermal protection handles it, same as always.

> **If you do accidentally use the machine during the run:** just say so. The
> project has a built-in quality gate that detects interference and flags the
> affected measurements, so nobody will be misled — we'd just ask you to run it
> again.

---

## Step 5 — run the benchmark

```bat
.venv\Scripts\python benchmark.py --samples 50
```

**Takes about 13–15 minutes.** It prints a table of timings as it goes. You'll
see lines mentioning clock speeds and things being "quotable" or "TOO-NOISY" —
that's the program checking its own measurement quality, and some rejections are
completely normal.

Near the end it prints something like:

```
wrote results\benchmark.json  (790.4s total)
clocks: 41/48 rows quotable
```

That's it — you're done.

**If you need to stop it:** Ctrl+C is completely safe at any point. Nothing keeps
running in the background. Just start over when convenient.

---

## Step 6 — send back one file

```
tritonkv\results\benchmark.json
```

It's about **4 MB**. Discord, email, Google Drive, WeTransfer — whatever's
easiest.

**That single file contains everything needed** — your GPU model, its cache size,
driver version, and every measurement. You won't be asked follow-up questions.

If you want to see what you're sending, it's plain text (JSON). There is nothing
personal in it: no filenames, no usernames, no system information beyond the GPU
and Python/driver versions.

---

## What this does to your computer — the honest list

| | |
|---|---|
| **Disk used** | ~3.2 GB in the `.venv` folder, plus ~1 GB in pip's download cache. |
| **Installed system-wide** | Nothing. |
| **Admin rights** | Not needed, not requested. |
| **Driver / GPU settings** | Untouched. The project explicitly cannot change GPU clock settings — that needs admin, and it doesn't try. |
| **Registry / startup items** | None. |
| **Background processes** | None. Nothing survives the run. |
| **Network use** | ~3 GB during install. At runtime it fetches one small model *configuration* file (a few KB — **not** model weights) and works fine offline if it can't reach the internet. |
| **Heat / fans** | Full GPU load for ~13 min. Like gaming. Fans get loud. |
| **Battery** | Should be plugged in. |

---

## Removing it afterwards

Delete the `tritonkv` folder. That's genuinely all of it.

To also clear pip's download cache (~1 GB):

```bat
.venv\Scripts\pip cache purge
```

(Run that *before* deleting the folder, or just use `python -m pip cache purge`.)

---

## Troubleshooting

### `python` is not recognised

Python isn't installed or isn't on PATH. Install from
<https://www.python.org/downloads/> and **tick "Add Python to PATH"** in the
installer. Then open a *new* terminal.

Try `py --version` too — on Windows the launcher is sometimes `py` rather than
`python`. If so, use `py -m venv .venv` in Step 2.

### `pip` can't find torch 2.12.0+cu130

Almost always a Python version problem. Check:

```bat
.venv\Scripts\python --version
```

If it's older than 3.10 or very new (3.14+), that build may not exist for it.
Install Python 3.12, delete the `.venv` folder, and redo Step 2.

### PyTorch can't see the GPU (`torch.cuda.is_available()` is `False`)

In order of likelihood:

1. **Driver too old.** CUDA 13 needs driver **580 or newer**. Check with
   `nvidia-smi`. Updating your GeForce driver fixes this most of the time.
2. **You installed the CPU-only build.** If you ran plain `pip install torch`
   without the `--index-url`, you got the CPU version. Fix:
   ```bat
   .venv\Scripts\pip uninstall -y torch
   .venv\Scripts\pip install torch==2.12.0+cu130 --index-url https://download.pytorch.org/whl/cu130
   ```
3. **You're on a laptop with switchable graphics** and the terminal is running on
   the integrated GPU. In NVIDIA Control Panel → Manage 3D Settings, set the
   preferred processor to the NVIDIA GPU.

If your driver can't be updated, tell Priyansh — there are PyTorch builds for
older CUDA versions and he can give you a different install command.

### `ModuleNotFoundError: No module named 'triton'`

On Windows the package is `triton-windows`, which `requirements.txt` handles — so
this usually means Step 2's second command didn't finish. Re-run:

```bat
.venv\Scripts\pip install -r requirements.txt
```

On Linux, make sure you changed that line to `triton==3.8.0` as described in
Step 2.

### The correctness test fails

Send Priyansh the full error text. Useful context to include:

```bat
.venv\Scripts\python -c "import torch,sys; print(sys.version); print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

**Don't run the benchmark if this fails.** The results wouldn't mean anything.

### It's extremely slow the first time / seems frozen

Normal. Triton compiles GPU code on demand, and the first compile of each variant
takes a while with no output. Give it 10 minutes before assuming it's stuck.

### Out of memory / `CUDA out of memory`

The benchmark deliberately allocates large buffers to exceed the GPU's cache. It
fits in 8 GB, but not alongside a game. Close everything else using the GPU and
retry. If it still fails, tell Priyansh your VRAM size — there's a smaller mode:

```bat
.venv\Scripts\python benchmark.py --quick
```

(That's a reduced run — only use it if the full one won't fit, and say so.)

### Antivirus blocks something

Some antivirus software dislikes the on-the-fly GPU compiler writing temporary
files. If you see permission errors, adding the project folder to your
antivirus's exclusions usually resolves it.

---

## FAQ

**Is this safe / what is it actually doing?**
It's an open-source benchmark. Everything is readable at
<https://github.com/PS12007/tritonkv>. It multiplies matrices on your GPU and
times how long that takes. It doesn't touch your files, network, or accounts.

**Will this damage my GPU?**
No. It's the same load as a game, for 13 minutes. Modern GPUs throttle themselves
long before anything harmful happens, and this doesn't and can't override that.

**Can I run it on battery?**
You can, but please don't — the results become useless because the GPU throttles.

**Can I use my laptop while it runs?**
Please don't, for the 13-minute benchmark. The setup and testing steps are fine.

**Do I need to install CUDA?**
No. The PyTorch package includes everything it needs. You only need the graphics
driver you already have.

**What if my results look "wrong" or lots of rows say TOO-NOISY?**
That's fine and expected — some rejections happen on every machine, including the
original. The program is designed to be honest about which of its own
measurements it trusts. Send the file regardless.

**Can I look at the results myself?**
Sure. After the run:
```bat
.venv\Scripts\python audit_claims.py
```
That writes `results/audit.md`, a plain-English list of every claim the project
makes and whether your data supports it.

---

## Optional: what the result actually tests

Only if you're curious.

When an AI model generates text it stores everything it has written so far in a
"KV cache". Long conversation, big cache — and the GPU spends most of its time
just fetching that cache from memory.

The obvious optimisation is to **compress it**: store each number in 4 bits
instead of 16. Four times smaller, four times fewer bytes to fetch, so it should
be faster. This project's kernel reads the compressed form directly, without
unpacking it first.

And it *is* much faster than the standard PyTorch path — 10× to 70×.

**But almost none of that speedup comes from the compression.** It comes from
splitting the work across the GPU's cores in a way PyTorch doesn't for this case.
Compared fairly — same split, compressed versus not — the compression usually
makes things *slower*, because unpacking costs more than the saved bytes are
worth.

Compression only wins once the cache is too large to fit in the GPU's fast
on-chip memory (its "L2"). Then, finally, fetching 4× fewer bytes pays.

**Which is exactly why your GPU matters.** That conclusion is a claim about L2
size, and it has only ever been measured on one card with one L2. Yours is
different. If the explanation is right, the crossover point — the conversation
length where compression starts winning — should move in a direction that can be
predicted in advance from your L2 size alone.

Your 15 minutes turns "this is true on one laptop" into "this is true about
GPUs". Thanks again.
