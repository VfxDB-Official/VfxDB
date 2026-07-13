# VfxDB

**A Visual Effects Volume Dataset and Benchmark for VDB-Native Generative Modeling.**

VfxDB is a large-scale VDB/OpenVDB volumetric effects dataset (smoke, fire, dust, explosions, …) together with a reproducible benchmark and a diffusion training/inference framework for **sparse 3D volume generation**. This repository contains the training and inference code plus thin compatibility launchers for the dataset-local downloader.

The models here are class-conditional (classifier-free guidance) 3D diffusion models that generate sparse voxel volumes derived from VDB grids, in two settings: **static** (single volume) and **temporal** (volume sequences). The framework uses an *Atomic-Continuous* prior to address the distribution mismatch between vanilla diffusion and the intrinsic sparsity of VDB data.

![VfxDB teaser visualization](https://vfxdb-official.github.io/VfxDB/static/images/main_image.jpg)

- 🌐 Project page: <https://vfxdb-official.github.io/VfxDB/>
- 📦 Dataset: <https://huggingface.co/datasets/ryogishiki/VfxDB>
- 📄 Paper: *coming soon*

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
> NVIDIA driver 570.158.01. The dataset downloader uses the pip-installed
> `zstandard` package and does not require a system `zstd` command. Native
> OpenVDB builds and maintainer packaging have separate system dependencies.

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
The dataset repository is also the **only source** for the downloader, Rich TUI,
specification, and downloader regression tests. This training repository keeps
only two thin compatibility entries, `tools/download_extract_data.py` and
`tools/download_extract_meta.py`, backed by one small bootstrap. They retrieve
the two source modules from the pinned dataset-tool commit through the standard
Hugging Face cache, then forward all arguments unchanged. Starting either entry
does not download dataset payloads.

Projects that do not use this training repository should obtain the tool
directly using the instructions in the
[VfxDB dataset README](https://huggingface.co/datasets/ryogishiki/VfxDB).

Every invocation first downloads all `<Category>/category_index.json` files and
fully installs the published `<Category>/index/*.json` files. These JSON files
are mandatory inputs to download planning and training, not an optional
download. A bare command prepares only this JSON control data and downloads no
VDB tar:

```bash
# Prepare all required JSON; no VDB data is downloaded
python tools/download_extract_data.py

# Presets
python tools/download_extract_data.py /data/vfxdb --preset smoke
python tools/download_extract_data.py /data/vfxdb --preset medium
python tools/download_extract_data.py /data/vfxdb --preset full

# One percentage across all categories
python tools/download_extract_data.py /data/vfxdb --percentage 10

# Target 1000 usable samples from each named category, rounded up to whole tars
python tools/download_extract_data.py /data/vfxdb \
  --category CloudWave \
  --category SurfaceFire \
  --max-samples 1000
```

For an interactive terminal workflow, add `--tui` without a selection option:

```bash
python tools/download_extract_data.py /data/vfxdb --tui
```

The TUI confirms the destination, prepares the same mandatory category indexes and
`<Category>/index/*.json` files, and only then offers modes using those installed local indexes.
Before any VDB tar starts, it shows the exact tar allocation, usable sample counts, conservative
network/install sizes, cache and destination free space, and whole-tar rounding. The review action
can download, go back and change the selection, or quit while keeping required JSON ready. Full
mode additionally requires the exact text `FULL`. Revision, cache location, and the explicit
diagnostic option to retain known IO-bad samples live under advanced settings.

During transfer the TUI separates overall tar progress, current-file bytes, JSON-file work, and
install status. `Ctrl-C` stops safely and the same selection can be rerun to continue. `--tui`
requires interactive stdin and stdout; scripts, pipes, and batch jobs should use the ordinary CLI
examples above.

The three presets are deterministic:

- **Smoke:** the first 2 complete tars from every category; a shorter category contributes all of
  its tars and its shortfall is not redistributed.
- **Medium:** 20% of all dataset tars, rounded up and allocated across categories in balanced
  rounds; categories with remaining tars fill the target after shorter categories are exhausted.
- **Full:** every tar from every category.

Preset mode cannot be combined with `--percentage`, `--category`, or `--max-samples`. Percentage
mode always covers all categories. Category mode requires both one or more `--category` arguments
and one shared positive `--max-samples`, applied separately to each category. If the last selected
tar crosses the requested maximum, it is downloaded in full; if the maximum exceeds the category,
that category is downloaded in full.

Selection, transfer, verification, and extraction are all whole-tar operations. Tar order follows
the first appearance of each sequence in the local category index. `EnvironmentalFog` remains
single-frame data—each VDB row counts as one sample—but it is still transferred as complete tars.

Known IO-bad files are handled after a complete tar is extracted. By default, the bad VDB and its
corresponding `<Category>/index/*.json` are removed, while the original category-index row is kept
and annotated with `"deleted_bad_io_sample": true`. With `--include-bad`, those files are retained
and the same row is annotated with `"deleted_bad_io_sample": false`. The option does not change
tar selection, ordering, quotas, or normal-sample counts.

The downloader pins the destination to one immutable Hugging Face commit, validates complete tars,
and uses `huggingface_hub` for bounded HTTP retries and its verified cache. Re-run the same command
after an interruption to reuse every completed tar and local installation; depending on the
installed Hub version, only the tar that was actively transferring may restart. Use `--cache-dir`
or `HF_HOME` to place the Hugging Face cache on another disk. Use `--revision` to choose a dataset
revision, and the standard `HTTPS_PROXY`
environment variable for a proxy. Start with an empty destination: a nonempty directory created by
an older or manual downloader has no pinned revision state and is rejected rather than silently
mixed with the current dataset commit. If an IO-bad policy transition is interrupted, rerun the
same downloader command; the training loader refuses that incomplete root until reconciliation
finishes. The complete behavioral contract lives with the dataset in
[`docs/DOWNLOADER_SPEC.md`](https://huggingface.co/datasets/ryogishiki/VfxDB/blob/main/docs/DOWNLOADER_SPEC.md).

For training or experiments that consume the per-sample JSON, set `return_meta: true` in the
training config. The loader then follows each `category_index.json` row's exact `meta_path` and
returns that decoded `<Category>/index/*.json` object as `sample["meta"]`; this also makes the JSON
mandatory for every installed VDB. Missing or invalid JSON fails explicitly instead of silently
substituting another sample. A DataLoader batch keeps these heterogeneous objects as a list under
`batch["meta"]` while tensor fields are collated normally. The released configs currently leave
`return_meta` disabled unless the experiment needs those fields.

Expected extracted layout:

```text
data/vdbset/
  SurfaceFire/
    category_index.json
    index/
      <sample>.json
    <sequence>/
      <sample>.vdb
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
> python tools/download_extract_data.py data/vdbset --preset full
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

### Project website

The website (`index.html`, `static/`) lives on the
[`gh-pages`](https://github.com/VfxDB-Official/VfxDB/tree/gh-pages) branch and is served by GitHub
Pages at <https://vfxdb-official.github.io/VfxDB/>. It is not part of the code branches.
