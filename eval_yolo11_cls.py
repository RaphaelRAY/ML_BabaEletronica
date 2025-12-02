"""
Avaliacao de modelo YOLO11 de classificacao em um dataset de pastas
<raiz>/train/<classe>, <raiz>/val|valid|validation/<classe>, <raiz>/test/<classe>.

Uso tipico:
python eval_yolo11_cls.py --weights runs/classify/yolo11-cls/weights/best.pt --data-root tcc-sleep-posture-classes.v4-teste-dataset-1.folder --split test
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

try:
    from ultralytics import YOLO
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "Dependencias ausentes. Instale com: pip install ultralytics matplotlib"
    ) from exc

# Caminho padrao do dataset
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parent / "tcc-sleep-posture-classes.v4-teste-dataset-1.folder"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avaliar modelo YOLO11 de classificacao")
    parser.add_argument(
        "--weights",
        required=True,
        help="Caminho para os pesos (.pt), ex.: runs/classify/yolo11-cls/weights/best.pt",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Raiz do dataset com pastas train/val(/test)",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "valid", "validation", "test"],
        help="Split a avaliar",
    )
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=224, help="Resolucao quadrada usada na avaliacao")
    parser.add_argument("--device", default="auto", help="Dispositivo (auto, cpu, 0, 0,1,2)")
    parser.add_argument("--project", default="runs/classify", help="Diretorio base para salvar os resultados")
    parser.add_argument("--name", default="val", help="Nome da subpasta do experimento")
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Usa apenas a pasta test/<classe> (sem precisar de train/val); calcula acuracia simples.",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        help="Caminho explicito para pasta test/<classe>. Se omitido, usa <data-root>/test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    if not data_root.exists():
        raise SystemExit(f"Dataset nao encontrado em {data_root}")
    weights_path = Path(args.weights).expanduser().resolve()
    if not weights_path.exists():
        raise SystemExit(f"Pesos nao encontrados em {weights_path}")

    device = args.device
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"

    model = YOLO(str(weights_path))

    if args.test_only:
        test_dir = args.test_path.resolve() if args.test_path else data_root / "test"
        if not test_dir.exists():
            raise SystemExit(f"Pasta de teste nao encontrada em {test_dir}")
        name_to_idx = {name: idx for idx, name in model.names.items()}
        idx_to_name = {idx: name for idx, name in model.names.items()}
        num_classes = len(idx_to_name)
        confusion = np.zeros((num_classes, num_classes), dtype=int)  # [pred, true]

        total = 0
        correct = 0
        per_class_total: Counter[str] = Counter()
        per_class_correct: Counter[str] = Counter()
        allowed_suffix = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for class_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
            label = class_dir.name
            if label not in name_to_idx:
                print(f"[AVISO] Classe '{label}' nao esta nos nomes do modelo {list(model.names.values())}, pulando.")
                continue
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() not in allowed_suffix:
                    continue
                res = model(img_path, imgsz=args.imgsz, device=device, verbose=False)[0]
                pred_idx = int(res.probs.top1)
                pred_name = model.names[pred_idx]
                total += 1
                per_class_total[label] += 1
                if pred_name == label:
                    correct += 1
                    per_class_correct[label] += 1
                true_idx = name_to_idx[label]
                confusion[pred_idx, true_idx] += 1  # linhas = predito, colunas = verdadeiro

        if total == 0:
            raise SystemExit("Nenhuma imagem valida encontrada em test.")

        overall_acc = correct / total
        print(f"Acuracia (test-only): {overall_acc:.4f} ({correct}/{total})")
        print("Por classe:")
        for cls in sorted(per_class_total.keys()):
            acc = per_class_correct[cls] / per_class_total[cls] if per_class_total[cls] else 0.0
            print(f"  {cls}: {acc:.4f} ({per_class_correct[cls]}/{per_class_total[cls]})")

        # Salva matriz de confusao em PNG e CSV
        save_dir = Path(args.project) / args.name
        save_dir.mkdir(parents=True, exist_ok=True)
        # CSV
        csv_path = save_dir / "confusion_matrix_test.csv"
        header = ",".join([idx_to_name[i] for i in range(num_classes)])  # colunas = verdadeiro
        with csv_path.open("w", encoding="utf-8") as f:
            f.write(",true->," + header + "\n")
            for i in range(num_classes):
                row = ",".join(str(confusion[i, j]) for j in range(num_classes))
                f.write(f"{idx_to_name[i]},{row}\n")  # linhas = predito
        # PNG heatmap
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(confusion, cmap="Blues")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels([idx_to_name[i] for i in range(num_classes)], rotation=45, ha="right")
        ax.set_yticklabels([idx_to_name[i] for i in range(num_classes)])
        ax.set_xlabel("Verdadeiro")
        ax.set_ylabel("Predito")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, int(confusion[i, j]), ha="center", va="center", color="black")
        fig.tight_layout()
        png_path = save_dir / "confusion_matrix_test.png"
        fig.savefig(png_path)
        plt.close(fig)
        print(f"Matriz de confusao salva em:\n  {png_path}\n  {csv_path}")

    else:
        metrics = model.val(
            data=str(data_root),
            split=args.split,
            batch=args.batch,
            imgsz=args.imgsz,
            device=device,
            project=args.project,
            name=args.name,
        )

        # Impressao amigavel dos principais metadados
        print("\nResumo de metrica:")
        for key in ("top1", "top5", "acc", "precision", "recall"):
            if hasattr(metrics, key):
                print(f"{key}: {getattr(metrics, key)}")
        if hasattr(metrics, "results_dict"):
            print("Resultados brutos:", metrics.results_dict)


if __name__ == "__main__":
    main()
