# Ortho_DGCNN_transformer

Deep learning pipeline for predicting **per-tooth orthodontic movement** (clear‑aligner staging) directly from **3D tooth point clouds**. Given a segmented lower-arch scan at the start of treatment, the model predicts the **cumulative 6‑degree‑of‑freedom transformation** (3 translation + 3 rotation components) each tooth undergoes by the end of treatment, along with which parameters are actively involved in the movement and their direction.

This is the modeling backbone for an orthodontic treatment-planning pipeline: instead of a technician manually staging tooth movements, the network learns to predict target tooth positions from the pre-treatment dental arch geometry.

## Table of contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Data format](#data-format)
- [Training](#training)
- [Inference](#inference)
- [Notes on the two pipelines](#notes-on-the-two-pipelines)
- [Acknowledgments](#acknowledgments)

## Overview

- **Input:** a pre-treatment lower-arch scan, segmented per tooth (14 mandibular teeth, FDI 31–37 / 41–47), as a point cloud sampled from each tooth's mesh.
- **Output, per tooth:**
  - Cumulative **translation magnitude** (Left/Right, Forward/Backward, Extrude/Intrude — mm)
  - Cumulative **rotation magnitude** (Buccal/Lingual, Mesial/Distal, Rotation — degrees)
  - **Active/inactive** classification per parameter (does this tooth actually move along this axis?)
  - **Direction** classification per parameter (positive vs. negative sign of the movement)
- **Backbone:** a DGCNN (Dynamic Graph CNN) point-cloud encoder feeding a Transformer/attention-augmented sequential decoder that reasons over the dental arch as an ordered sequence of teeth.

The current, actively developed pipeline lives in [`OrthoDGCNN_v1.8/`](./OrthoDGCNN_v1.8) and predicts the **cumulative (final) transformation** per tooth in a single shot. An earlier, per-stage pipeline (predicting a full sequence of intermediate aligner stages) lives at the repository root — see [Notes on the two pipelines](#notes-on-the-two-pipelines).

## Architecture

```
Per-tooth point clouds (14 × N × C)
        │
        ▼
   DGCNN encoder            (models/DGCNN.py)
   EdgeConv ×4 + graph max-pooling → 1 embedding per tooth
        │
        ▼
 Arch-order sorting          sort teeth by mesial→distal (x-coordinate) position
        │
        ▼
  LayerNorm + neighbor        (models/OrthoDGCNN.py)
  attention context           (models/GRU_cumulativedecoder.py)
        │
        ▼
 Sequential per-tooth LSTM decoder, tooth-by-tooth over the arch,
 with teacher forcing on ground-truth cumulative transforms during training
        │
        ├── Translation head  →  3 values (mm)
        ├── Rotation head     →  3 values (degrees)
        ├── Active head       →  6 sigmoid logits (per-parameter activity)
        └── Direction head    →  6 sigmoid logits (per-parameter sign)
```

Key design choices:

- **DGCNN encoder** (`models/DGCNN.py`): builds a k-NN graph over each tooth's points and applies 4 stacked EdgeConv blocks, concatenating multi-scale features before a final 1D conv + max-pool to produce one embedding per tooth. A **PointNet++** encoder (`models/pointnet2.py`) is also available as a drop-in alternative via `--encoder_type`.
- **Arch ordering**: tooth embeddings are re-sorted by their point-cloud centroid along the arch axis so the decoder processes teeth in anatomical order (e.g. 37→36→…→31→41→…→47), then un-sorted back to FDI order for the output.
- **Neighbor attention**: before decoding, each tooth's feature is contextualized with its immediate left/right neighbors (multi-head attention) blended with the global arch embedding, since tooth movement is influenced by adjacent teeth.
- **Sequential decoder with teacher forcing** (`models/GRU_cumulativedecoder.py`): an LSTM steps through the 14 teeth in arch order, at each step embedding the previous tooth's (ground-truth, with a decaying probability, or predicted) transformation and feeding it back in — encouraging the model to learn arch-level dependencies between teeth.
- **Four prediction heads**: separate MLPs for translation magnitude, rotation magnitude, per-parameter active/inactive classification, and per-parameter direction classification. Regression and classification heads can be frozen independently (`--is_freeze`) to support staged training.
- **Loss** (`losses_cumulative.py`): Huber loss on translation/rotation magnitudes (masked to active parameters only), weighted binary cross-entropy for the active and direction heads (with per-parameter positive-class weighting for imbalance), plus per-parameter F1 tracking.

## Repository structure

```
.
├── README.md
├── dataset.py                     # legacy per-stage dataset (JawTeethDataset)
├── train.py                       # legacy per-stage training entry point
├── inference.py                   # legacy per-stage inference entry point
├── models/                        # legacy per-stage model components
│   ├── DGCNN.py
│   ├── OrthoDGCNN.py
│   ├── StagePredictor.py
│   └── StageTransformer.py
│
└── OrthoDGCNN_v1.8/                # current cumulative-transformation pipeline
    ├── dataset.py                  # JawTeethDataset + CumulativeJawTeethDataset
    ├── train_cumulative.py         # training entry point
    ├── inference_cumulative.py     # inference entry point
    ├── losses_cumulative.py        # Huber + weighted BCE losses, F1 tracking
    ├── optimizers.py               # optimizer/scheduler factory
    └── models/
        ├── DGCNN.py                # point-cloud encoder (EdgeConv-based)
        ├── pointnet2.py            # alternative point-cloud encoder
        ├── pointnet2_utils.py
        ├── OrthoDGCNN.py           # top-level model: encoder + decoder
        ├── GRU_cumulativedecoder.py  # active sequential decoder (used by OrthoDGCNN.py)
        ├── GRUDecoder.py           # alternative GRU-based decoder
        ├── TransformerDecoder.py   # alternative Transformer-based decoder
        ├── OrthoDGCNN_decoder.py   # alternative model wiring
        ├── CumulativeTransformationModel.py  # alternative Transformer encoder/head variant
        └── perteeth.py             # alternative per-tooth transformer decoder
```

`models/GRU_cumulativedecoder.py` (`GRUToothDecoder`) is the decoder wired into `OrthoDGCNN.py` and used by `train_cumulative.py` / `inference_cumulative.py`. The other decoder/model files under `models/` are alternative architectures explored during development and are not wired into the main training script by default.

## Installation

```bash
git clone https://github.com/ossmjm/Ortho_DGCNN_transformer.git
cd Ortho_DGCNN_transformer/OrthoDGCNN_v1.8
```

Dependencies (no `requirements.txt` is checked in yet — install the following):

```bash
pip install torch numpy pandas scikit-learn openpyxl trimesh open3d
pip install pytorch_optimizer torch_optimizer lion-pytorch
```

- `torch` — model + training loop (CUDA recommended)
- `pandas` / `openpyxl` — reading `.xlsx` transformation labels
- `scikit-learn` — train/val splitting, feature scaling
- `trimesh` — mesh loading (legacy pipeline)
- `open3d` — curvature-aware point sampling (current pipeline)
- `pytorch_optimizer`, `torch_optimizer`, `lion_pytorch` — optional optimizers (Adan, RAdam, Lion) exposed via `optimizers.py`

## Data format

Each case is a numbered folder under `--data_dir`:

```
Data/
├── 1/
│   ├── ori/
│   │   └── before_treatment.json      # per-tooth mesh: {"teeth": {"31": {"v": [[x,y,z], ...], "f": [...]}, ...}}
│   └── cumulative_transformations.xlsx
├── 2/
│   ├── ori/
│   │   └── before_treatment.json
│   └── cumulative_transformations.xlsx
...
```

`cumulative_transformations.xlsx` columns (one row per tooth per case):

| Column | Meaning |
|---|---|
| `Jaw_ID` | case folder name |
| `Tooth_ID` | FDI tooth number (31–37, 41–47) |
| `Left/Right (mm` | translation |
| `Forward/Backward (mm)` | translation |
| `Extrude/Intrude (mm)` | translation |
| `Buccal/Lingual (degrees)` | rotation |
| `Mesial/Distal (degrees)` | rotation |
| `Rotation (degrees)` | rotation |

Only the mandibular arch (14 teeth) is currently supported. Missing teeth in the JSON are zero-filled; missing rows in the Excel file default that tooth's transformation to zero (inactive).

Processed cases are cached as pickles under `--cache_dir` to avoid re-parsing meshes/Excel files on every run.

## Training

From `OrthoDGCNN_v1.8/`:

```bash
python train_cumulative.py \
  --data_dir ./Data \
  --output_dir ./output \
  --cache_dir ./cache \
  --encoder_type dgcnn \
  --num_points 256 \
  --channels 3 \
  --embed_dim 256 \
  --batch_size 4 \
  --epochs 100 \
  --lr 5e-5 \
  --optimizer_name adamw \
  --scheduler_name cosineannealing
```

Useful flags:

- `--encoder_type {dgcnn, pointnet2}` — point-cloud backbone
- `--scaler_type {robust, standard}` / `--use_scaler` — normalization of transformation targets (in addition to the fixed translation/rotation ranges baked into the dataset)
- `--is_freeze {none, regression, classification}` — freeze the opposite task's heads for staged training
- `--optimizer_name {adamw, radam, lion, sparseadam, adan}`, `--scheduler_name {cosineannealing, reduceonplateau, linear}`
- `--patience` — early stopping on validation loss

Checkpoints (`model_epoch_N.pth`, `best_model.pth`, `last_model.pth`, `best_model_final.pth`) and per-epoch loss/F1 history (`.npy`) are written to `--output_dir`.

## Inference

```bash
python inference_cumulative.py \
  --data_dir ./Data \
  --model_path ./output/best_model.pth \
  --output_dir ./output \
  --cache_dir ./cache \
  --embed_dim 256 \
  --num_points 256 \
  --channels 3
```

Produces `predicted_cumulative_transformations.xlsx` in `--output_dir`, with one row per (case, tooth) containing the predicted translation/rotation values in the same column layout as the training labels.

## Notes on the two pipelines

| | Root (`dataset.py`, `train.py`, `inference.py`) | `OrthoDGCNN_v1.8/` |
|---|---|---|
| Target | Per-stage transformations across up to `max_stages` aligner steps | Single cumulative (final) transformation per tooth |
| Label file | `Transformations.xlsx` with a `Stage` column + `num_stages.xlsx` | `cumulative_transformations.xlsx` |
| Decoder | `StageTransformer` (stage-sequence transformer) | `GRUToothDecoder` (arch-sequence LSTM with teacher forcing) |
| Status | Earlier iteration, kept for reference | Actively developed / current entry point |

New work should build on `OrthoDGCNN_v1.8/`.

## Acknowledgments

The point-cloud encoder is based on **Dynamic Graph CNN (DGCNN)** (Wang et al., *Dynamic Graph CNN for Learning on Point Clouds*), and the alternative encoder is based on **PointNet++** (Qi et al.). The sequential decoder combines this with attention/Transformer-style components to model dependencies between teeth across the dental arch.
