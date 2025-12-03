# Babá Eletrônica (Baby Monitor) - ML Sleep Posture Classification

This project implements a machine learning system for classifying baby sleep postures using YOLO11 image classification. It detects and classifies different baby sleeping positions to assist in monitoring infant safety.

## Overview

The system classifies baby sleep postures into the following categories:
- **left** - Baby sleeping on left side
- **prone** - Baby sleeping face down (on tummy)
- **sunpine** (supine) - Baby sleeping face up (on back)
- **absent** - No baby present in the frame

## Requirements

- Python 3.8+
- PyTorch
- Ultralytics (YOLO11)
- Matplotlib (for evaluation visualization)
- NumPy

### Installation

```bash
pip install ultralytics matplotlib numpy
```

## Project Structure

```
ML_BabaEletronica/
├── train_yolo11_cls.py    # Training script for YOLO11 classification
├── eval_yolo11_cls.py     # Evaluation script for trained models
├── preproc.py             # Dataset preprocessing/splitting script
├── yolo11n-cls.pt         # Pre-trained YOLO11 classification weights
├── yolo11n.pt             # Pre-trained YOLO11 detection weights
├── baby_sleep_balanced/   # Balanced dataset with train/val/test splits
│   ├── train/
│   ├── val/
│   └── test/
├── dataset-boneca/        # Additional test dataset
│   └── data.yaml          # Dataset configuration
├── runs/                  # Training and evaluation results
└── tcc-sleep-posture-classes.v4-teste-dataset-1.folder/  # Main dataset
```

## Dataset Structure

The datasets follow the standard image classification folder structure:
```
dataset_root/
├── train/
│   ├── absent/
│   ├── left/
│   ├── prone/
│   └── sunpine/
├── valid/
│   ├── absent/
│   ├── left/
│   ├── prone/
│   └── sunpine/
└── test/
    ├── absent/
    ├── left/
    ├── prone/
    └── sunpine/
```

## Usage

### Training

Train a YOLO11 classification model:

```bash
python train_yolo11_cls.py \
    --data-root tcc-sleep-posture-classes.v4-teste-dataset-1.folder \
    --weights yolo11n-cls.pt \
    --epochs 100 \
    --batch 32 \
    --imgsz 224
```

#### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-root` | `tcc-sleep-posture-classes.v4-...` | Path to dataset root |
| `--weights` | `yolo11n-cls.pt` | Initial weights file |
| `--epochs` | 100 | Number of training epochs |
| `--batch` | 32 | Batch size |
| `--imgsz` | 224 | Image resolution (square) |
| `--workers` | 8 | Number of dataloader workers |
| `--device` | auto | Device (auto, cpu, 0, 0,1,2) |
| `--project` | `runs/classify` | Results directory |
| `--name` | `yolo11-cls` | Experiment name |
| `--patience` | 20 | Early stopping patience |
| `--lr0` | 0.01 | Initial learning rate |

### Evaluation

Evaluate a trained model on validation or test data:

```bash
python eval_yolo11_cls.py \
    --weights runs/classify/yolo11-cls/weights/best.pt \
    --data-root tcc-sleep-posture-classes.v4-teste-dataset-1.folder \
    --split test
```

For test-only evaluation (without train/val structure):

```bash
python eval_yolo11_cls.py \
    --weights runs/classify/yolo11-cls/weights/best.pt \
    --test-only \
    --test-path path/to/test/folder
```

#### Evaluation Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--weights` | (required) | Path to trained model weights |
| `--data-root` | `tcc-sleep-posture-classes.v4-...` | Path to dataset root |
| `--split` | val | Split to evaluate (val, valid, validation, test) |
| `--batch` | 32 | Batch size |
| `--imgsz` | 224 | Image resolution |
| `--device` | auto | Device selection |
| `--test-only` | False | Evaluate on test folder only |
| `--test-path` | None | Explicit path to test folder |

### Dataset Preprocessing

Use the preprocessing script to split data into train/valid/test (80/10/10):

```bash
python preproc.py
```

**Note:** Edit the `base` path variable in the script to point to your dataset location.

## Results

Training and evaluation results are saved in the `runs/classify/` directory, including:
- Trained model weights (`best.pt`, `last.pt`)
- Training metrics and logs
- Confusion matrices (PNG and CSV formats)
- Accuracy metrics per class

## Model

This project uses YOLO11 (Ultralytics) for image classification. The model is initialized with pre-trained weights (`yolo11n-cls.pt`) and fine-tuned on the baby sleep posture dataset.

## License

This project was developed as part of a TCC (Trabalho de Conclusão de Curso - Final Year Project).
