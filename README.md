# VfxDB

Training and inference code for sparse 3D volume generation on OpenVDB visual-effects data.

This repository is the runnable code. The dataset, paper weights, and project page live elsewhere.

- 🌐 Project page: <https://vfxdb-official.github.io/VfxDB/>
- 📄 Paper: <https://dl.acm.org/doi/10.1145/3799902.3811178>
- 📦 Dataset: <https://huggingface.co/datasets/ryogishiki/VfxDB>
- 🧠 Checkpoints: <https://huggingface.co/ryogishiki/VfxDB-models>

![VfxDB teaser visualization](https://vfxdb-official.github.io/VfxDB/static/images/main_image.jpg)

Scripts you will actually run: `sh/` (train / smoke tests), `infer_one_stage_hf.py` (inference), `tools/download_extract_data.py` (data). YAML lives in `configs/`.

## Quick start

Tested on Linux + NVIDIA GPU, Python 3.10, PyTorch 2.5.1+cu124. The commands below use tiny dummy data and do **not** download the dataset.

```bash
git clone https://github.com/VfxDB-Official/VfxDB.git
cd VfxDB

conda create -n vfxdb python=3.10 -y && conda activate vfxdb
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-core.txt

curl -L https://github.com/VfxDB-Official/VfxDB/releases/download/prebuilt-v1/vdb_ext-runtime-linux-x86_64-py310.tar.gz \
  | tar -xz -C "$(python -c 'import site; print(site.getsitepackages()[0])')"

./sh/train_dummy_static.sh
./sh/validate_pipeline_parity.sh
./sh/infer_one_stage_hf.sh \
  --ckpt runs/dummy_static/one_stage_dummy_static/ckpt/step_000002 \
  --out-dir results/dummy_infer
```

If those three scripts succeed, the environment is ready.

The prebuilt VDB runtime (`openvdb` + `vdb_ext`) is Linux x86_64 / CPython 3.10. Other platforms: build [vdb_ext](https://github.com/ghosard/vdb_ext) from source.

## Download a slice of real data

```bash
python tools/download_extract_data.py data/vdbset --preset smoke
```

`smoke` is the small starter set (two sequence tars per category). For a bigger slice, the full 4.6 TB, or one category, use the [dataset README](https://huggingface.co/datasets/ryogishiki/VfxDB).

## Train

```bash
./sh/run_train_cond_static_32.sh   --data-root data/vdbset   # class-conditional, static
./sh/run_train_cond_temporal_32.sh --data-root data/vdbset   # class-conditional, sequence
./sh/run_train_uncond_static_32.sh --data-root data/vdbset   # unconditional, static
```

Configs are in `configs/`. Logs and checkpoints go to `runs/<setting>/<experiment>/`.

`NUM_PROCESSES` is the GPU count for `accelerate launch`. `batch_size` in the YAML is per GPU:

```bash
NUM_PROCESSES=8 MIXED_PRECISION=fp16 ./sh/run_train_cond_static_32.sh --data-root data/vdbset
```

The released configs cap training at `max_train_samples: 200000`. Set that to `0` in the YAML to use every sample on disk.

## Inference

The three paper EMA checkpoints load straight from Hugging Face. No manual weight download.

```bash
python infer_one_stage_hf.py --config configs/infer_paper_static_unconditional_32.yaml
python infer_one_stage_hf.py --config configs/infer_paper_static_conditional_32.yaml
python infer_one_stage_hf.py --config configs/infer_paper_temporal_conditional_32.yaml
```

Override any config key on the command line, for example `--eval-cfg-class-id 3 --out-dir results/infer/class_3`.

## Citation

```bibtex
@inproceedings{VfxDB,
author = {Shu, Junwei and Liu, Hantang and Miao, Dawei and Song, Wenzheng and Yuan, Mingyang and Liu, Wenjie and Chen, Changgu and Li, Yang and Wang, Changbo},
title = {VfxDB: A Visual Effects Volume Dataset and Benchmark for VDB-Native Generative Modeling},
year = {2026},
isbn = {9798400725548},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3799902.3811178},
doi = {10.1145/3799902.3811178},
booktitle = {Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers},
articleno = {195},
numpages = {11},
keywords = {VDB, Visual Effects, Diffusion Generative Model, Machine Learning},
location = {
},
series = {SIGGRAPH Conference Papers '26}
}
```

Dataset and checkpoints are [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
