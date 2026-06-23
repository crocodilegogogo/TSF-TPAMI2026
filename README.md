# Triple Spectral Fusion for Sensor-based Human Activity Recognition

### Good News! Our paper has been accepted by `IEEE Transactions on Pattern Analysis and Machine Intelligence`.

This repository contains the PyTorch implementation of 'Triple Spectral Fusion For Sensor-based Human Activity Recognition'.

![framework](Figs/Overrall_Framework.jpg)

## Algorithm Introduction:

This paper is an extension of our previous conference version [(paper link)](https://dl.acm.org/doi/pdf/10.1145/3534584) in which we proposed an If-ConvTransformer framework for HAR. The additional contributions as compared to our preliminary version are listed as follows:

- We design a modality node fusion block via the adaptive filtering mechanism in graph Fourier domain. This block fuses the homogeneous and heterogeneous  modality information, and thus performs multi-sensor  fusion more effectively than IF-ConvTransformer.

- We construct a temporal information fusion block via  the adaptive wavelet frequency selection mechanism. This block effectively suppresses temporal redundancies, which improves IF-ConvTransformer in terms of  context correlations and computation efficiency.
- We propose a Triple Spectral Fusion (TSF) framework  via specific filtering mechanisms in three spectral domains. Our TSF framework achieves state-of-the-art  performance on ten public HAR datasets.

## Project structure:

```text
.
├── Figs/                         # Paper figures used by README
├── datasets/                     # Put downloaded/preprocessed datasets here
├── src/
│   ├── main.py                   # Main train/test entry
│   ├── classifiers/              # Model implementations
│   ├── saved_models/             # Runtime checkpoint output directory
│   └── utils/
│       ├── constants.py          # CLI and dataset/model factory settings
│       ├── hf_downloader.py      # Hugging Face dataset/model auto-download helpers
│       ├── hyperparams.yaml      # Dataset/model hyperparameters
│       └── load_*_dataset/       # Dataset loaders and preprocessing code
├── requirements.txt
├── environment.yml
└── pyproject.toml
```

## Environment:

Training and testing are performed on a PC with Ubuntu 22.04 system, 2 x NVIDIA 3090 GPUs.

Important dependency packages are provided in 'requirements.txt' file.

Recommended Conda setup:

```bash
conda env create -f environment.yml
conda activate TSF
```

If DGL installation fails through `environment.yml`, install the CUDA 11.6 compatible DGL wheel manually according to your CUDA/PyTorch environment, then rerun the project.

## Automatic data and model preparation

**1. Datasets and pretrained TSF checkpoints can be resolved automatically from Hugging Face:**

- dataset repo: https://huggingface.co/datasets/crocodilegogogo/TSF-Datasets/tree/main
- model repo: https://huggingface.co/crocodilegogogo/TSF-Models/tree/main

The download logic is implemented in `src/utils/hf_downloader.py` using the official `huggingface_hub` package. It is called from the data loading entry and, in `TEST` mode, from the model setup step.

By default, missing datasets are downloaded before data loading. Missing TSF checkpoints are downloaded automatically in `TEST` mode. To disable either behavior:

```bash
python -m src.main --PATTERN TRAIN --DATASETS HAPT --CLASSIFIERS TSF_torch --no-auto-download-data

python -m src.main --PATTERN TEST --DATASETS HAPT --CLASSIFIERS TSF_torch --no-auto-download-models
```

**2. Expected local dataset folders remain compatible with the original loaders:**

```text
datasets/UCI HAPT/HAPT_Dataset/
datasets/Motion-Sense/
datasets/HHAR/Per_subject_npy/
datasets/MobiAct/Per_subject_no_NED_npy/
datasets/Opportunity/
datasets/Pamap2/
datasets/RealWorld/
datasets/DSADS/
datasets/SHO/
```

**3. Expected TSF model folders are:**

```text
src/saved_models/<DATASET>/TSF_torch/SUBJECT_<ID>/best_validation_model.pkl
src/saved_models/MobiAct/TSF_torch/FOLD_<ID>/best_validation_model.pkl
```

**4. Optional Hugging Face controls:**

The downloaded .zip files are cached under the following  paths:

```text
datasets/.hf_cache/              # dataset repo snapshots/cache
src/saved_models/.hf_cache/      # model repo snapshots/cache
```

**5. Original Datasets and data preprocessing:**

If you do not want to use the processed datasets, the downloading links of the original datasets are also provided in the top parts of our data loading code. The code for data preprocessing of each dataset are provided in the 'utils' folder.

## Training

Run commands from the repository root:

```bash
python -m src.main \
  --PATTERN TRAIN \
  --DATASETS HAPT \
  --CLASSIFIERS TSF_torch \
  --INFERENCE_DEVICE TEST_CUDA \
  --seed 6
```

You can pass multiple datasets or classifiers:

```bash
python -m src.main \
  --PATTERN TRAIN \
  --DATASETS HAPT Opportunity \
  --CLASSIFIERS IF_ConvTransformer_torch TSF_torch \
  --INFERENCE_DEVICE TEST_CUDA
```

## Testing

After checkpoints are available under `src/saved_models/`, run:

```bash
python -m src.main \
  --PATTERN TEST \
  --DATASETS HAPT Opportunity \
  --CLASSIFIERS TSF_torch \
  --INFERENCE_DEVICE TEST_CUDA
```

For CPU-only inference:

```bash
python -m src.main \
  --PATTERN TEST \
  --DATASETS HAPT \
  --CLASSIFIERS TSF_torch \
  --INFERENCE_DEVICE TEST_CPU
```

## GPU selection

Set CUDA visibility externally:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.main --PATTERN TRAIN --DATASETS HAPT --CLASSIFIERS TSF_torch
```

## Hyperparameters

Dataset-level and classifier-level hyperparameters are stored in:

```text
src/utils/hyperparams.yaml
```

## Results

![Results](Figs/Results.jpg)

## Contact

**Welcome to raise issues or email to zhangy2658@mail.sysu.edu.cn or yundazhangye@163.com for any question regarding this work.**