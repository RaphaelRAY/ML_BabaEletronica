import os
import shutil
import random

# Caminho base do dataset
base = r"C:\Users\rapha\Downloads\BEBE\Treino\tcc-sleep-posture-classes.v4-teste-dataset-1.folder"

# Pastas de saída
splits = ["train", "valid", "test"]
for split in splits:
    os.makedirs(os.path.join(base, split), exist_ok=True)

# Obtém todas as classes (pastas) que estão diretamente dentro do base
classes = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)) and d not in splits]

print("Classes encontradas:", classes)

# Para cada classe, realizar o split 80/10/10
for cls in classes:
    cls_path = os.path.join(base, cls)
    images = os.listdir(cls_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * 0.8)
    valid_end = int(total * 0.9)

    train_files = images[:train_end]
    valid_files = images[train_end:valid_end]
    test_files = images[valid_end:]

    # Cria pastas para cada split
    for split in splits:
        dst = os.path.join(base, split, cls)
        os.makedirs(dst, exist_ok=True)

    # Move arquivos
    for file in train_files:
        shutil.move(os.path.join(cls_path, file), os.path.join(base, "train", cls, file))

    for file in valid_files:
        shutil.move(os.path.join(cls_path, file), os.path.join(base, "valid", cls, file))

    for file in test_files:
        shutil.move(os.path.join(cls_path, file), os.path.join(base, "test", cls, file))

    # Remove pasta original da classe
    shutil.rmtree(cls_path)

print("Divisão 80/10/10 concluída com sucesso!")
