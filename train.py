import os
import random
import shutil
import subprocess
import sys

# =========================
# SETTINGS
# =========================

DATASET_FILE = "train.txt"   # Raw dataset file
TRAIN_RATIO = 0.9
EPOCHS = 50

TRAIN_DATA_DIR = "./train_data"
IMG_DIR = "./train_data/plate_dataset/images"

TRAIN_FILE = "train_v5.txt"
VAL_FILE = "val_v5.txt"

CONFIG = "configs/rec/PP-OCRv5/en_PP-OCRv5_mobile_rec.yml"

USE_GPU = True

# =========================
# CHECK PADDLEOCR STRUCTURE
# =========================

if not os.path.exists("tools/train.py"):
    raise Exception("❌ Run this script inside PaddleOCR folder")

if not os.path.exists(DATASET_FILE):
    raise Exception(f"❌ Dataset file not found: {DATASET_FILE}")

if not os.path.exists(CONFIG):
    raise Exception("❌ PPOCR v5 config not found. Update PaddleOCR repo.")

print("✅ Files verified")

# =========================
# CREATE FOLDERS
# =========================

os.makedirs(IMG_DIR, exist_ok=True)

# =========================
# CLEAN DATA + COPY IMAGES
# =========================

clean_lines = []

print("🔧 Preparing dataset...")

with open(DATASET_FILE, "r") as f:
    for line in f:
        parts = line.strip().split()

        if len(parts) >= 2:
            img_path = parts[0]
            label = parts[-1]

            if os.path.exists(img_path):

                new_img = os.path.join(
                    IMG_DIR,
                    os.path.basename(img_path)
                )

                if not os.path.exists(new_img):
                    shutil.copy(img_path, new_img)

                rel_path = f"plate_dataset/images/{os.path.basename(img_path)}"
                clean_lines.append(f"{rel_path}\t{label}\n")

print(f"✅ Valid samples: {len(clean_lines)}")

# =========================
# SHUFFLE + SPLIT
# =========================

random.shuffle(clean_lines)

split = int(len(clean_lines) * TRAIN_RATIO)

train_lines = clean_lines[:split]
val_lines = clean_lines[split:]

with open(TRAIN_FILE, "w") as f:
    f.writelines(train_lines)

with open(VAL_FILE, "w") as f:
    f.writelines(val_lines)

print(f"✅ Train samples: {len(train_lines)}")
print(f"✅ Val samples: {len(val_lines)}")

# =========================
# TRAIN COMMAND
# =========================

python_exec = sys.executable

cmd = [
    python_exec,
    "tools/train.py",
    "-c", CONFIG,
    "-o",
    f"Global.epoch_num={EPOCHS}",
    f"Global.use_gpu={str(USE_GPU)}",
    "Global.save_model_dir=./output/v5_model",
    "Train.dataset.data_dir=./train_data",
    "Eval.dataset.data_dir=./train_data",
    f'Train.dataset.label_file_list=["{TRAIN_FILE}"]',
    f'Eval.dataset.label_file_list=["{VAL_FILE}"]'
]

print("\n🚀 Starting PPOCR V5 Training...\n")

subprocess.run(cmd, check=True)

print("\n🎉 TRAINING COMPLETE!")
print("📁 Model saved in: output/v5_model")

