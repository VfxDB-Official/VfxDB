# VfxDB

Official repository for the VfxDB paper:

**VfxDB: A Visual Effects Volume Dataset and Benchmark for VDB-Native Generative Modeling**

![VfxDB teaser visualization](static/images/main_image.jpg)

This repository contains the project website assets and the minimal Hugging Face diffusers training, inference, and validation path extracted from `vgrad_train`.

## Install

```bash
pip install -r requirements-core.txt
```

For CUDA, install the matching PyTorch wheel for your machine first, then install the remaining requirements.

## Dummy Smoke Test

```bash
python tools/make_dummy_vdbset.py --out dummy_data/vdbset --num-categories 2 --num-sequences 2 --num-frames 4 --size 8
python train_one_stage.py --config configs/train_dummy_static.yaml
python tests/test_pipeline_parity.py
python infer_one_stage_hf.py --config configs/infer_one_stage_hf.yaml --ckpt runs/dummy_static/one_stage_dummy_static/ckpt/step_000002 --out-dir results/dummy_infer
```

The production training wrappers are in `sh/`. They assume you run from this folder.

## Milestone Correspondence

The paper/milestone training entry was:

```bash
python train_one_stage.py --config configs/conditional_temporal_32_baseline_5ws.yaml
```

The open-source temporal wrapper keeps the same training path and HF/diffusers trainer, with public paths and no-render HF eval:

```bash
./sh/run_train_one_stage.sh --data-root /path/to/VDBSet
```

The static 32 baseline wrapper is:

```bash
./sh/run_train_one_stage_acc.sh --data-root /path/to/VDBSet
```

Both configs retain the paper-line one-stage settings such as class conditioning, occupancy head, EMA, `log1p` value space, DDPM schedule, P2 weighting, and temporal previous-frame conditioning for the temporal model.

## Dataset Layout

Training expects a root like:

```text
root/
  dataset_index.json
  CategoryA/
    category_index.json
    seq000/
      sample__n0000.npz
      sample__n0000.json
```

Each `.npz` file must contain `vol` with shape `[D, H, W]`. Optional `occ`, `leaf_base`, and `lvl_sizes` keys are supported.
