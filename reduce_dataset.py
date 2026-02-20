import os
import random
import shutil

random.seed(42)

BASE = r"data raw"
LIMITS = {
    "train": 800,
    "validation": 200
}

for split in ["train", "validation"]:
    for cls in ["real", "fake"]:
        path = os.path.join(BASE, split, cls)

        files = [f for f in os.listdir(path) if f.lower().endswith((".jpg",".png",".mp4",".jpeg"))]
        random.shuffle(files)

        keep = files[:LIMITS[split]]

        print(f"{split}/{cls} -> garder {len(keep)} fichiers")

        # créer dossier réduit
        new_path = os.path.join(BASE, split + "_mini", cls)
        os.makedirs(new_path, exist_ok=True)

        for f in keep:
            shutil.copy(
                os.path.join(path, f),
                os.path.join(new_path, f)
            )
import os

print(len(os.listdir("data raw/train_mini/real")))
print(len(os.listdir("data raw/train_mini/fake")))
