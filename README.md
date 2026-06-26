# VfxDB

**A Visual Effects Volume Dataset and Benchmark for VDB-Native Generative Modeling.**

VfxDB is a large-scale VDB/OpenVDB volumetric effects dataset (smoke, fire, dust, explosions, …) together with a reproducible benchmark and a diffusion training/inference framework for **sparse 3D volume generation**. This repository contains the training and inference code and the data download tooling.

The models here are class-conditional (classifier-free guidance) 3D diffusion models that generate sparse voxel volumes derived from VDB grids, in two settings: **static** (single volume) and **temporal** (volume sequences). The framework uses an *Atomic-Continuous* prior to address the distribution mismatch between vanilla diffusion and the intrinsic sparsity of VDB data.

![VfxDB teaser visualization](https://vfxdb-official.github.io/VfxDB/static/images/main_image.jpg)

- 🌐 Project page: <https://vfxdb-official.github.io/VfxDB/>
- 📦 Dataset: <https://huggingface.co/datasets/ryogishiki/VfxDB>
- 📄 Paper: *coming soon*

TODO:
-[x] Dataset Release: <https://huggingface.co/datasets/ryogishiki/VfxDB>
-[x] Training/Inference Code
-[ ] Pretrained Checkpoints
-[ ] Better dataset integration 

## Contents

- [Quick Start](#quick-start)
- [What's in this repo](#whats-in-this-repo)
- [Installation](#installation)
- [Download data](#download-data)
- [Training](#training)
- [Inference](#inference)
- [Troubleshooting & advanced](#troubleshooting--advanced)

## Quick Start

The fastest way to confirm a working setup. The smoke test below generates tiny dummy
VDB data locally (no dataset download required) and runs a 2-step train → validate → infer cycle.

```bash
# 1. Clone
git clone https://github.com/VfxDB-Official/VfxDB.git
cd VfxDB

# 2. Create the environment (Python 3.10) and install dependencies
conda create -n vfxdb python=3.10 -y && conda activate vfxdb
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-core.txt

# 3. Download & install the prebuilt VDB runtime (Linux x86_64 / CPython 3.10)
curl -L https://github.com/VfxDB-Official/VfxDB/releases/download/prebuilt-v1/vdb_ext-runtime-linux-x86_64-py310.tar.gz \
  | tar -xz -C "$(python -c 'import site; print(site.getsitepackages()[0])')"

# 4. Smoke test: no data download needed
./sh/train_dummy_static.sh
./sh/validate_pipeline_parity.sh
./sh/infer_one_stage_hf.sh \
  --ckpt runs/dummy_static/one_stage_dummy_static/ckpt/step_000002 \
  --out-dir results/dummy_infer
```

If all three commands succeed, your environment is ready. See [Installation](#installation) for
details (including the source build for non-Linux / non-py310 platforms), then
[Download data](#download-data) and [Training](#training) for real runs.

## What's in this repo

| Path | Purpose |
| --- | --- |
| `train_one_stage.py` | Config-driven training entrypoint. |
| `infer_one_stage_hf.py` | Inference from Hugging Face-format checkpoints. |
| `configs/` | Training and inference YAML configs. |
| `sh/` | Runnable shell wrappers for the released settings. |
| `tools/` | Dataset download, extraction, metadata, and dummy-data helpers. |
| `requirements-core.txt` | Python packages required for training and inference. |
| `requirements-render.txt` | Optional packages for richer visual evaluation/rendering. |

> The project website source lives on the [`gh-pages`](https://github.com/VfxDB-Official/VfxDB/tree/gh-pages)
> branch, not here — cloning the code does not pull down the site assets.

## Installation

> Tested on Linux + NVIDIA GPU, Python 3.10.16, PyTorch 2.5.1+cu124 (CUDA runtime 12.4),
> NVIDIA driver 570.158.01. System tools used by the scripts: `git`, `tar`, `zstd`.

### 1. Python environment and PyTorch

```bash
conda create -n vfxdb python=3.10 -y && conda activate vfxdb
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

### 2. Python dependencies

```bash
python -m pip install -r requirements-core.txt
# Optional rendering/evaluation extras (Mitsuba, PyVista, matplotlib, imageio):
python -m pip install -r requirements-render.txt
```

### 3. VDB runtime (`openvdb` + `vdb_ext`)

The public `.vdb` data is loaded through `openvdb` and `vdb_ext`, which must be importable in the
same Python environment. On **Linux x86_64 / CPython 3.10**, download the prebuilt bundle from the
[releases page](https://github.com/VfxDB-Official/VfxDB/releases/tag/prebuilt-v1) — it contains the
extension modules plus their OpenVDB/TBB/Boost runtime libraries, with RPATH set for same-directory
loading — and extract it into your `site-packages`:

```bash
curl -L https://github.com/VfxDB-Official/VfxDB/releases/download/prebuilt-v1/vdb_ext-runtime-linux-x86_64-py310.tar.gz \
  | tar -xz -C "$(python -c 'import site; print(site.getsitepackages()[0])')"
```

On other platforms, build `vdb_ext` from source — see
[Building `vdb_ext` from source](#building-vdb_ext-from-source).

### Verify the installation

```bash
python - <<'PY'
import sys, torch, openvdb, vdb_ext
print("python", sys.version.split()[0])
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| available", torch.cuda.is_available())
print("openvdb", openvdb.__file__)
print("vdb_ext", vdb_ext.__file__)
PY
```

## Download data

The public dataset is hosted at <https://huggingface.co/datasets/ryogishiki/VfxDB>.
The downloader fetches `dataset_index.json`, the selected `category_index.json` files, the needed
VDB tar archives, and (by default) a small metadata archive. The released training configs do not
require metadata (`require_meta: false`).

By default the downloader is intentionally conservative: with no selection flags it pulls a **single
category** (`CloudWave`) capped at **1000 samples per category** — enough to try a run without
accidentally downloading the whole (large) dataset. Use the flags below to control the scope.

```bash
# Small real VDB smoke dataset (24 SurfaceFire samples)
python tools/download_extract_data.py --data-root data/vdbset --folders SurfaceFire:24

# 1000-VDB example (CloudWave)
python tools/download_extract_data.py --data-root data/vdbset \
  --categories CloudWave --max-samples-per-category 1000

# Full dataset: every category, every sample (large — see --plan-only first)
python tools/download_extract_data.py --data-root data/vdbset --all

# Preview what --all would fetch (categories + archive count) without downloading:
python tools/download_extract_data.py --data-root data/vdbset --all --plan-only

# Behind a local proxy: add --proxy http://127.0.0.1:7890
# Skip metadata (indexes + VDB only): add --skip-meta
# Metadata only:
python tools/download_extract_meta.py --data-root data/vdbset
```

> `--all` is shorthand for `--all-categories --max-samples-per-category 0` (where `0` means "no
> per-category cap"). To scope to specific categories at full depth, combine
> `--categories A B --max-samples-per-category 0`.

Expected extracted layout:

```text
data/vdbset/
  dataset_index.json
  SurfaceFire/
    category_index.json
    24/
      _sequence_manifest.json
      s0024__n0000.vdb
      s0024__n0001.vdb
      ...
```

The dataset loader also supports `.npz` samples containing a `vol` array with shape `[D, H, W]`.

## Training

Run commands from the repository root. Command-line overrides use YAML keys; hyphens are accepted
in place of underscores (e.g. `--data-root` overrides `data_root`). Checkpoints and logs are written
under `runs/<setting>/<experiment>/`.

Released training wrappers (`*_static_32` / `*_temporal_32` use `accelerate launch`):

```bash
./sh/run_train_cond_static_32.sh   --data-root data/vdbset   # class-conditional, static
./sh/run_train_cond_temporal_32.sh --data-root data/vdbset   # class-conditional, sequence
./sh/run_train_uncond_static_32.sh --data-root data/vdbset   # unconditional, static
```

Control process count and mixed precision via environment variables. `NUM_PROCESSES` sets the number
of (GPU) processes for `accelerate launch`; `batch_size` in the config is **per process**, so the
effective global batch is `batch_size × NUM_PROCESSES` (the trainer does not split batches across
processes):

```bash
# 8-GPU launch; per-process batch 8 → global batch 64
NUM_PROCESSES=8 MIXED_PRECISION=fp16 ./sh/run_train_cond_static_32.sh --data-root data/vdbset
```

> **Training on the full dataset.** Training is fully config-driven. The configs cap the training
> set at `max_train_samples: 200000` (applied after the train/val/test split), so downloading more
> data does not by itself train on more. To use every available sample, set `max_train_samples: 0`
> (`0` = no cap) in the config you are running, e.g. `configs/train_static_32.yaml`. Then:
>
> ```bash
> python tools/download_extract_data.py --data-root data/vdbset --all
> NUM_PROCESSES=8 MIXED_PRECISION=fp16 ./sh/run_train_cond_static_32.sh --data-root data/vdbset
> ```
>
> Resume an interrupted run by pointing `--resume` at a checkpoint directory under
> `runs/<setting>/<experiment>/ckpt/`.

Short SurfaceFire smoke run (1 step):

```bash
./sh/run_train_cond_static_32.sh \
  --data-root data/vdbset \
  --categories "[SurfaceFire]" \
  --train-ratio 1 --val-ratio 0 --test-ratio 0 \
  --max-train-samples 120 --train-steps 1 --batch-size 1 --num-workers 0 \
  --eval-every 1000000 --ckpt-every 1000000 --log-every 1
```

1000-VDB CloudWave run:

```bash
./sh/run_train_cond_static_32.sh --data-root data/vdbset \
  --categories "[CloudWave]" --max-train-samples 1000
```

To skip automatic visual evaluation (e.g. when the rendering extras are not installed), disable it
and run inference manually from checkpoints:

```bash
./sh/run_train_cond_static_32.sh --data-root data/vdbset --eval-every 0
```

## Inference

Run inference from a Hugging Face-format checkpoint directory. Outputs are saved as `.npz` files
plus a `summary.json`.

```bash
./sh/infer_one_stage_hf.sh \
  --ckpt /path/to/hf/checkpoint \
  --out-dir results/infer \
  --eval-num-samples 2 \
  --sampling-steps 20
```

For conditional checkpoints, select a class by id when needed:

```bash
./sh/infer_one_stage_hf.sh \
  --ckpt /path/to/hf/checkpoint \
  --out-dir results/infer \
  --eval-cfg-class-id 0
```

## Troubleshooting & advanced

### Building `vdb_ext` from source

Use this path if you are **not** on Linux x86_64 with CPython 3.10. Source repo:
<https://github.com/ghosard/vdb_ext>.

Native dependencies: OpenVDB C++ headers/libs, NanoVDB headers (`nanovdb/NanoVDB.h`), oneTBB,
Boost.Iostreams, Blosc, zlib, xz/liblzma, LZ4, Snappy, Zstandard. Build dependencies:
`cmake>=3.24`, `ninja`, `scikit-build-core>=0.10`, `pybind11>=2.12`, `numpy>=1.23`
(plus CMake, Ninja, and a C++ compiler).

A conda-forge starting point for the native packages:

```bash
conda install -c conda-forge \
  cmake ninja cxx-compiler openvdb tbb boost blosc zlib xz lz4 snappy zstd
```

Then build and install into the active environment:

```bash
git clone https://github.com/ghosard/vdb_ext.git /tmp/vdb_ext
python -m pip install -r /tmp/vdb_ext/requirements-build.txt
python -m pip install /tmp/vdb_ext \
  --config-settings=cmake.define.VDB_EXT_OPENVDB_ROOT="$CONDA_PREFIX" \
  --config-settings=cmake.define.VDB_EXT_NANOVDB_ROOT=/path/to/openvdb/nanovdb \
  --config-settings=cmake.define.VDB_EXT_RPATH_DIRS="$CONDA_PREFIX/lib"
```

Notes:

- Depending on how OpenVDB is packaged, NanoVDB headers may need to come from an OpenVDB source
  checkout or a separate install prefix. If your OpenVDB install already exposes
  `nanovdb/NanoVDB.h`, point `VDB_EXT_NANOVDB_ROOT` at that same prefix.
- Make sure the OpenVDB runtime libraries are visible when importing `vdb_ext` (via
  `LD_LIBRARY_PATH`, RPATH, or the system loader cache).

### `ImportError` for `openvdb` / `vdb_ext`

Re-run the [verify step](#verify-the-installation). The most common cause is that the prebuilt
`.so` files were copied to the wrong `site-packages`, or the OpenVDB runtime libraries are not on
the loader path. Confirm the active interpreter is the `vfxdb` env (`which python`).

