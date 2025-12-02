"""
Treinamento de classificacao com YOLO11 usando Ultralytics.

O script espera um dataset no formato de pastas:
<raiz>/train/<classe>/*.jpg
<raiz>/(val|valid)/<classe>/*.jpg
<raiz>/test/<classe>/*.jpg  (opcional)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

try:
    from ultralytics import YOLO
    import torch
except ImportError as exc:  # Ultralytics ou torch nao instalados
    raise SystemExit(
        "Dependencias ausentes. Instale com: pip install ultralytics"
    ) from exc

# Ajuste o caminho padrao para a raiz do dataset conforme necessidade.
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parent / "tcc-sleep-posture-classes.v4-teste-dataset-1.folder"


def find_split(root: Path, candidates: tuple[str, ...]) -> Path:
    """Retorna o primeiro split existente dentre os nomes candidatos."""
    for name in candidates:
        candidate = root / name
        if candidate.exists():
            return candidate
    names = ", ".join(candidates)
    raise FileNotFoundError(f"Nao encontrei nenhum split ({names}) dentro de {root}")


def list_classes(split_dir: Path) -> list[str]:
    """Lista classes presentes em um split."""
    return sorted(
        d.name for d in split_dir.iterdir() if d.is_dir()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treinar YOLO11 para classificacao")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Caminho para a pasta raiz contendo train/val(/test)",
    )
    parser.add_argument(
        "--weights",
        default="yolo11n-cls.pt",
        help="Pesos iniciais (ex.: yolo11s-cls.pt, yolo11s-cls.pt ou caminho local)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Numero de epocas")
    parser.add_argument("--batch", type=int, default=32, help="Tamanho do batch")
    parser.add_argument("--imgsz", type=int, default=224, help="Resolucao quadrada das imagens")
    parser.add_argument("--workers", type=int, default=8, help="Numero de workers para dataloader")
    parser.add_argument("--device", default="auto", help="Dispositivo (auto, cpu, 0, 0,1,2)")
    parser.add_argument("--project", default="runs/classify", help="Diretorio base para os resultados")
    parser.add_argument("--name", default="yolo11-cls", help="Nome do experimento")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--lr0", type=float, default=0.01, help="Learning rate inicial")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset nao encontrado em {data_root}")

    # Resolve dispositivo: se auto e nao houver GPU, cai para cpu.
    device = args.device
    if args.device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"

    train_dir = find_split(data_root, ("train",))
    val_dir = find_split(data_root, ("val", "validation", "valid"))
    try:
        test_dir = find_split(data_root, ("test",))
    except FileNotFoundError:
        test_dir = None

    classes = list_classes(train_dir)
    print(f"Dataset: {data_root}")
    print(f"Train dir: {train_dir}")
    print(f"Val dir: {val_dir}")
    if test_dir:
        print(f"Test dir: {test_dir}")
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Device selecionado: {device}")

    model = YOLO(args.weights)
    model.train(
        data=str(data_root),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        lr0=args.lr0,
    )


if __name__ == "__main__":
    main()
