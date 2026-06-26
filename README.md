# VfxDB

VfxDB training and inference code for VDB-native diffusion models, plus the project website assets.

![VfxDB teaser visualization](static/images/main_image.jpg)

## Repository Layout

- `train_one_stage.py`: config-driven training entrypoint.
- `infer_one_stage_hf.py`: inference from Hugging Face-format checkpoints.
- `configs/`: training and inference YAML configs.
- `sh/`: runnable shell wrappers for the released training settings.
- `tools/`: dataset download, extraction, metadata, and dummy-data helpers.
- `requirements-core.txt`: Python packages required for training and inference.
- `requirements-render.txt`: optional packages for richer visual evaluation/rendering.
- `index.html` and `static/`: project website files.

## Environment

Tested environment:

- Linux with an NVIDIA GPU
- Python 3.10.16
- PyTorch 2.5.1+cu124
- PyTorch CUDA runtime 12.4 (`cu124`)
- NVIDIA driver 570.158.01; `nvidia-smi` reports CUDA 12.8

System tools used by the scripts:

- `git`
- `tar`
- `zstd`
- CMake, Ninja, and a C++ compiler if building `vdb_ext` from source

The public `.vdb` data is loaded through `openvdb` and `vdb_ext`, which must be importable in the same Python environment. This repository includes a prebuilt Linux x86_64 / CPython 3.10 bundle under `prebuilt/linux-x86_64-py310/`. The bundle contains the Python extension modules plus their OpenVDB/TBB/Boost runtime libraries, with RPATH set for same-directory loading.

Use the source build path below if you are not on Linux x86_64 with CPython 3.10.

`vdb_ext` source repository:

<https://github.com/ghosard/vdb_ext>

`vdb_ext` native dependencies:

- OpenVDB C++ headers and libraries
- NanoVDB headers, including `nanovdb/NanoVDB.h`
- oneTBB
- Boost.Iostreams
- Blosc
- zlib
- xz / liblzma
- LZ4
- Snappy
- Zstandard

`vdb_ext` Python/build dependencies:

- `cmake>=3.24`
- `ninja`
- `scikit-build-core>=0.10`
- `pybind11>=2.12`
- `numpy>=1.23`

A conda-forge starting point for the native packages is:

```bash
conda install -c conda-forge \
  cmake ninja cxx-compiler openvdb tbb boost blosc zlib xz lz4 snappy zstd
```

Depending on how OpenVDB is packaged on your system, NanoVDB headers may still need to come from an OpenVDB source checkout or a separate install prefix.

Make sure the OpenVDB runtime libraries are visible when importing `vdb_ext`, for example through `LD_LIBRARY_PATH`, RPATH, or the system loader cache.

## Install

Clone this repository:

```bash
git clone https://github.com/VfxDB-Official/VfxDB.git
cd VfxDB
```

Create and activate a Python 3.10 environment:

```bash
conda create -n vfxdb python=3.10
conda activate vfxdb
python -m pip install --upgrade pip
```

Install PyTorch. The tested wheel family is `cu124`:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

Install the core Python requirements:

```bash
python -m pip install -r requirements-core.txt
```

Optional rendering/evaluation extras:

```bash
python -m pip install -r requirements-render.txt
```

Install the prebuilt `openvdb` / `vdb_ext` bundle:

```bash
SITE_PACKAGES="$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
cp prebuilt/linux-x86_64-py310/*.so* "$SITE_PACKAGES"/

python - <<'PY'
import openvdb
import vdb_ext
print("openvdb", openvdb.__file__)
print("vdb_ext", vdb_ext.__file__)
PY
```

Or build `vdb_ext` from source in the same environment:

```bash
git clone https://github.com/ghosard/vdb_ext.git /tmp/vdb_ext
python -m pip install -r /tmp/vdb_ext/requirements-build.txt
python -m pip install /tmp/vdb_ext \
  --config-settings=cmake.define.VDB_EXT_OPENVDB_ROOT=/path/to/openvdb/prefix \
  --config-settings=cmake.define.VDB_EXT_NANOVDB_ROOT=/path/to/openvdb/nanovdb \
  --config-settings=cmake.define.VDB_EXT_RPATH_DIRS=/path/to/openvdb/prefix/lib
```

If OpenVDB is installed into the active conda environment and NanoVDB headers are available from an OpenVDB source checkout, the install usually looks like:

```bash
python -m pip install /tmp/vdb_ext \
  --config-settings=cmake.define.VDB_EXT_OPENVDB_ROOT="$CONDA_PREFIX" \
  --config-settings=cmake.define.VDB_EXT_NANOVDB_ROOT=/path/to/openvdb/nanovdb \
  --config-settings=cmake.define.VDB_EXT_RPATH_DIRS="$CONDA_PREFIX/lib"
```

If your OpenVDB install already exposes `nanovdb/NanoVDB.h` under its include path, `VDB_EXT_NANOVDB_ROOT` can point to that same install prefix.

Check the environment:

```bash
python - <<'PY'
import sys
import torch
import openvdb
import vdb_ext

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("openvdb", openvdb.__file__)
print("vdb_ext", vdb_ext.__file__)
PY
```

## Download Data

The public dataset files are hosted at:

<https://huggingface.co/datasets/ryogishiki/VfxDB>

The downloader fetches `dataset_index.json`, selected `category_index.json` files, the needed VDB tar archives, and the metadata archive by default. The released training configs do not require metadata (`require_meta: false`), but the archive is small and is kept with the extracted dataset by default.

Small real VDB smoke dataset:

```bash
python tools/download_extract_data.py \
  --data-root data/vdbset \
  --folders SurfaceFire:24
```

1000-VDB example:

```bash
python tools/download_extract_data.py \
  --data-root data/vdbset \
  --categories CloudWave \
  --max-samples-per-category 1000
```

With a local proxy:

```bash
python tools/download_extract_data.py \
  --data-root data/vdbset \
  --categories CloudWave \
  --max-samples-per-category 1000 \
  --proxy http://127.0.0.1:7890
```

Skip metadata if you only want indexes and VDB archives:

```bash
python tools/download_extract_data.py \
  --data-root data/vdbset \
  --categories CloudWave \
  --max-samples-per-category 1000 \
  --skip-meta
```

Download metadata only:

```bash
python tools/download_extract_meta.py --data-root data/vdbset
```

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

## Train

Run commands from the repository root. Command-line overrides use YAML keys with hyphens accepted as underscores, for example `--data-root` overrides `data_root`.

Released training wrappers:

```bash
./sh/run_train_cond_static_32.sh --data-root data/vdbset
./sh/run_train_cond_temporal_32.sh --data-root data/vdbset
./sh/run_train_uncond_static_32.sh --data-root data/vdbset
```

`run_train_cond_static_32.sh` and `run_train_uncond_static_32.sh` use `accelerate launch`. You can set process count and mixed precision through environment variables:

```bash
NUM_PROCESSES=1 MIXED_PRECISION=fp16 ./sh/run_train_cond_static_32.sh --data-root data/vdbset
```

Short SurfaceFire smoke run:

```bash
./sh/run_train_cond_static_32.sh \
  --data-root data/vdbset \
  --categories "[SurfaceFire]" \
  --train-ratio 1 \
  --val-ratio 0 \
  --test-ratio 0 \
  --max-train-samples 120 \
  --train-steps 1 \
  --batch-size 1 \
  --num-workers 0 \
  --eval-every 1000000 \
  --ckpt-every 1000000 \
  --log-every 1
```

1000-VDB CloudWave run:

```bash
./sh/run_train_cond_static_32.sh \
  --data-root data/vdbset \
  --categories "[CloudWave]" \
  --max-train-samples 1000
```

For a minimal environment without optional visual evaluation dependencies, disable automatic evaluation and run inference manually from checkpoints:

```bash
./sh/run_train_cond_static_32.sh \
  --data-root data/vdbset \
  --eval-every 0
```

Checkpoints and logs are written under the configured `runs/<setting>/<experiment>/` directory.

## Inference

Run inference from a Hugging Face-format checkpoint directory:

```bash
./sh/infer_one_stage_hf.sh \
  --ckpt /path/to/hf/checkpoint \
  --out-dir results/infer \
  --eval-num-samples 2 \
  --sampling-steps 20
```

For conditional checkpoints, set a class by id when needed:

```bash
./sh/infer_one_stage_hf.sh \
  --ckpt /path/to/hf/checkpoint \
  --out-dir results/infer \
  --eval-cfg-class-id 0
```

Outputs are saved as `.npz` files plus a `summary.json`.

## Dummy Smoke Test

This path does not require downloading public VDB archives:

```bash
./sh/train_dummy_static.sh
./sh/validate_pipeline_parity.sh
./sh/infer_one_stage_hf.sh \
  --ckpt runs/dummy_static/one_stage_dummy_static/ckpt/step_000002 \
  --out-dir results/dummy_infer
```
