# Caries-DETR

**Tooth Structure-aware Prior and Lesion-aware Dynamic Loss Refinement for DETR Based Caries Detection**

## Abstract

As dental caries appear as subtle, low-contrast lesions in intraoral imaging, existing deep learning models face significant challenges in the early detection of caries. While recent Transformer-based detectors have shown promising results in natural images, they often fail to capture the domain-specific anatomical priors crucial for dental caries detection. In this paper, we propose **Caries-DETR**, a specialized Transformer framework for caries detection in intraoral images. A **Tooth Structure-aware Query Initialization (TSQI)** is designed, leveraging large-scale intraoral photograph pre-training and a Structure Perception Branch (SPB) to integrate high-frequency structural priors, guiding the model to focus on anatomically significant lesion areas. Furthermore, we design a **Lesion-aware Dynamic Loss Refinement (LDLR)** to implement quality-driven hard mining through adaptive loss reweighting based on lesion size, anatomical relevance, and prediction quality, optimizing detection for subtle lesions. Extensive experiments on two public intraoral photograph datasets (i.e., AlphaDent with 1,455 images across 9 categories, and DentalAI with 2,495 images across 4 categories) demonstrate that Caries-DETR achieves state-of-the-art performance compared to existing methods and exhibits good generalization and robustness.

---

## Architecture Overview

```
Intraoral Image
       |
       v
 ResNet-50 Backbone
       |
       v
 ChannelMapper Neck (4-scale FPN)
       |
       +-------------------------------+
       v                               v
 Deformable DINO Encoder    Structure Perception Branch (SPB)
       |                      (lightweight CNN heatmap from
       |                       large-scale pre-training)
       v                               v
 DINO Two-Stage Proposals    TSQIQueryGenerator
       |                      (top-k structure-guided
       |                       query initialization)
       +---------------+---------------+
                       v
               DINO Decoder (6 layers)
                       |
                       v
                CariesDETRHead
                 +-- Classification head
                 +-- Regression head
                 +-- Lesion-aware Dynamic Loss Refiner (LDLR)
                      +-- w_cls  = 1 + alpha * (1 - cls_prob)
                      +-- w_bbox = 1 + beta  * (1 - GIoU_quality)
                      +-- w_iou  = 1 + gamma * (1 - IoU)
```

### Key Components

| Module | File | Description |
|---|---|---|
| `StructurePerceptionBranch` | `caries_detr/models/tsqi_module.py` | Lightweight CNN producing a structural prior heatmap from large-scale pre-trained features, highlighting anatomically significant regions such as tooth boundaries and lesion margins |
| `TSQIQueryGenerator` | `caries_detr/models/tsqi_module.py` | Selects the top-k spatially salient positions guided by the SPB heatmap and projects them into DINO-compatible content queries and positional encodings |
| `LesionDynamicLossRefiner` | `caries_detr/models/ldlr_module.py` | Computes three independent per-prediction quality-aware weights (IoU, classification, bbox quality) for adaptive loss reweighting, implementing quality-driven hard mining for subtle lesions |
| `CariesDETRHead` | `caries_detr/models/caries_detr_head.py` | Integrates TSQI and LDLR into the DINO detection head (MMDetection-compatible registered module) |
| `GradientStructureGenerator` | `tools/train_lipp.py` | Scharr-gradient-based pseudo GT generator for self-supervised SPB pre-training (LIPP) |
| `AlphaDentDataset` | `caries_detr/datasets/alphadent_dataset.py` | COCO-format dataset loader for the 9-class AlphaDent intraoral photograph dataset (converted from YOLO instance segmentation format) |
| `DentalAIDataset` | `caries_detr/datasets/dentalai_dataset.py` | COCO-format dataset loader for the 4-class DentalAI intraoral photograph dataset (2,495 images, 28,904 annotated objects) |

---

## Installation

**Prerequisites:** Python >= 3.8, CUDA >= 11.1.

```bash
# 1. Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2. Install MMEngine and MMCV
pip install mmengine
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

# 3. Install MMDetection
pip install mmdet

# 4. Install remaining requirements
pip install -r requirements.txt

# 5. (Optional) Install in editable mode
pip install -e .
```

---

## Dataset Preparation

Caries-DETR is evaluated on two public datasets. Both must be converted to the standard COCO JSON annotation format before training.

### AlphaDent (9-class intraoral photographs)

**Source:** [Kaggle — Alpha Dent Competition](https://www.kaggle.com/competitions/alpha-dent/data)

AlphaDent comprises **1,455 intraoral photographs** with pixel-level instance segmentation annotations across 9 fine-grained categories, covering common dental restorations (Abrasion, Filling, Crown) and a detailed classification of caries severity and location (Caries 1 to Caries 6). We split the dataset into training (1,237 images), validation (83 images), and testing (135 images) subsets.

The original dataset is distributed in [Ultralytics/YOLO instance segmentation format](https://docs.ultralytics.com/) (.jpg images + .txt label files + `yolo_seg_train.yaml`). Convert the annotations to COCO JSON format before use (bounding boxes can be derived from the instance segmentation masks).

Expected directory layout after conversion:

```
data/AlphaDent/
+-- train2017/
+-- val2017/
+-- annotations/
    +-- instances_train2017.json
    +-- instances_val2017.json
```

Classes (9): `Abrasion`, `Filling`, `Crown`, `Caries 1 class` -- `Caries 6 class`

### DentalAI (4-class intraoral photographs)

**Source:** [Dataset Ninja — DentalAI](https://datasetninja.com/dentalai)

DentalAI consists of **2,495 intraoral photographs** collected from diverse clinical scenarios, with pixel-level instance segmentation annotations across 4 categories (Tooth, Caries, Cavity, and Crack). We split the dataset into training (1,991 images), validation (254 images), and testing (250 images) subsets.

Convert the instance segmentation annotations to COCO JSON bounding-box format before use. Expected directory layout:

```
data/DentalAI/
+-- train2017/
+-- val2017/
+-- annotations/
    +-- instances_train2017.json
    +-- instances_val2017.json
```

Classes (4): `Tooth`, `Caries`, `Cavity`, `Crack`

---

## SPB Pre-training (LIPP)

The Structure Perception Branch (SPB) is pre-trained via **Large-scale Intraoral Photograph Pre-training (LIPP)** on unlabelled intraoral photographs.  The pretext task trains the SPB to regress a Scharr-gradient-based structural heatmap from frozen backbone features, enabling it to learn high-frequency anatomical priors (tooth boundaries, enamel edges, fissure patterns) *without* any bounding-box annotations.

### Step 1 — Prepare unlabelled intraoral photographs

Collect a large set of intraoral photographs and place them under a single root directory (sub-directories are supported):

```
/path/to/intraoral_photos/
+-- patient_001/
|   +-- img_001.jpg
|   +-- img_002.jpg
+-- patient_002/
|   +-- ...
+-- ...
```

No annotation files are needed.

### Step 2 — Run LIPP pre-training

```bash
python tools/train_lipp.py \
    --config configs/caries_detr_alphadent.py \
    --data-root /path/to/intraoral_photos/ \
    --save-dir work_dirs/lipp_pretrain \
    --epochs 50 --batch-size 16 --lr 1e-4 \
    --gpus 0,1,2,3
```

Key arguments:

| Argument | Description |
|---|---|
| `--config` | Any Caries-DETR config (used to build the backbone + neck architecture) |
| `--data-root` | Root directory of unlabelled intraoral images |
| `--save-dir` | Output directory for SPB checkpoints |
| `--img-size` | Input resolution (default: `800 800`) |
| `--gpus` | Comma-separated GPU ids for `DataParallel` |

Outputs in `--save-dir`:
- `spb_pretrain_epoch_<N>.pth` — per-epoch checkpoints
- `spb_pretrain_best.pth` — best checkpoint (smoothed loss)
- `spb_pretrain_final.pth` — final epoch checkpoint

### Step 3 — Load pre-trained SPB into Caries-DETR

Set `init_cfg` in your config file to load the pre-trained SPB weights:

```python
bbox_head=dict(
    type='CariesDETRHead',
    ...
    init_cfg=dict(
        type='Pretrained',
        checkpoint='work_dirs/lipp_pretrain/spb_pretrain_best.pth',
        prefix='structure_perception_branch.',
    ),
)
```

Then proceed to the standard training procedure below.

---

## Training

### Single GPU

```bash
python tools/train.py configs/caries_detr_alphadent.py
```

### Multi-GPU (e.g., 4 GPUs)

```bash
bash tools/dist_train.sh configs/caries_detr_alphadent.py 4
```

---

## Evaluation

```bash
python tools/test.py configs/caries_detr_alphadent.py checkpoints/best_model.pth
```

---

## Configuration Reference

The two configuration files (`configs/caries_detr_alphadent.py` and `configs/caries_detr_dentalai.py`) expose the following key hyperparameters:

```python
# Tooth Structure-aware Query Initialization (TSQI)
tsqi_cfg = dict(
    in_channels=256,    # feature channels from the neck
    mid_channels=128,   # SPB hidden channels
    num_queries=900,    # total DINO queries
    query_dim=256,      # query embedding dimension
    topk=300,           # number of structure-guided positions to select
)

# Lesion-aware Dynamic Loss Refinement (LDLR)
ldlr_cfg = dict(
    gamma_iou=1.0,      # sensitivity for IoU-based weight
    alpha_cls=1.0,      # sensitivity for classification-based weight
    beta_bbox=1.0,      # sensitivity for bbox-quality-based weight
    enable_ldlr=True,   # set False to disable LDLR at runtime
)
```

---

## Project Structure

```
caries-detr/
+-- caries_detr/
|   +-- models/
|   |   +-- tsqi_module.py            # StructurePerceptionBranch + TSQIQueryGenerator
|   |   +-- ldlr_module.py            # LesionDynamicLossRefiner (LDLR)
|   |   +-- caries_detr_head.py       # CariesDETRHead (MMDet-registered)
|   +-- datasets/
|       +-- alphadent_dataset.py      # AlphaDentDataset (9 classes)
|       +-- dentalai_dataset.py       # DentalAIDataset (4 classes)
+-- configs/
|   +-- caries_detr_alphadent.py      # AlphaDent config
|   +-- caries_detr_dentalai.py       # DentalAI config
+-- tools/
|   +-- train.py                      # Main training script
|   +-- test.py                       # Evaluation script
|   +-- train_lipp.py                 # LIPP: SPB self-supervised pre-training
|   +-- dist_train.sh                 # Multi-GPU launcher
+-- requirements.txt
+-- setup.py
+-- LICENSE
+-- README.md
```

---

## Citation

If you use Caries-DETR in your research, please cite:

```bibtex
@article{caries-detr2025,
  title   = {Tooth Structure-aware Prior and Lesion-aware Dynamic Loss
             Refinement for {DETR} Based Caries Detection},
  author  = {Xuefen Liu and others},
  year    = {2025},
}
```

---

## Acknowledgements

This project is built upon:

- [DINO: DETR with Improved DeNoising Anchor Boxes](https://arxiv.org/abs/2203.03605)
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [Deformable DETR](https://arxiv.org/abs/2010.04159)

---

## License

This project is released under the [Apache 2.0 License](LICENSE).
