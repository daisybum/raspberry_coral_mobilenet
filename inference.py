#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, json, logging
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm

from pycoral.utils import edgetpu
from pycoral.adapters import common, classify

# logger initialization
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ─── 0. Edge TPU detection ─────────────────────────────────────
devices = edgetpu.list_edge_tpus()
using_tpu = bool(devices)

if using_tpu:
    logger.info(f"✅ Edge TPU detected: {devices} – running on TPU.")
else:
    logger.warning("⚠️ No Edge TPU detected – running in CPU mode.")

# ─── 1. Path configuration ─────────────────────────────────────
IMAGE_DIR = Path('/workspace/testset')
MODEL_PATH = Path('models/model_quant_edgetpu.tflite')
TOP_K = 1

logger.info("Path configuration completed.")

# ─── 2. Interpreter initialization ─────────────────────────────
if using_tpu:
    interpreter = edgetpu.make_interpreter(str(MODEL_PATH), device="usb")
else:
    from tflite_runtime.interpreter import Interpreter
    interpreter = Interpreter(model_path=str(MODEL_PATH))

interpreter.allocate_tensors()

# Check loaded delegates (may be a private attribute)
loaded_delegates = getattr(interpreter, "_delegates", None)
if loaded_delegates:
    logger.info(f"Loaded delegate: {loaded_delegates}")
else:
    logger.info("No delegates loaded (CPU mode)")

input_width, input_height = common.input_size(interpreter)
logger.info(f"Model input size: {input_width}×{input_height} RGB")

# ─── 3. Load class labels (from folder names) ────────────────
class_to_index = {
    cls: idx
    for idx, cls in enumerate(sorted([d.name for d in IMAGE_DIR.iterdir() if d.is_dir()]))
}
index_to_class = {v: k for k, v in class_to_index.items()}

logger.info(f"Class labels loaded: {json.dumps(index_to_class, ensure_ascii=False)}")

# ─── 4. Helper functions: preprocess & infer ─────────────────
def preprocess(img_path):
    img = Image.open(img_path).convert('RGB').resize(
        (input_width, input_height), Image.BILINEAR
    )
    return np.asarray(img).astype(np.uint8)

def infer(image_np):
    common.set_input(interpreter, image_np)
    interpreter.invoke()
    classes = classify.get_classes(interpreter, top_k=TOP_K)
    return [(c.id, c.score) for c in classes]

# ─── 5. Iterate over all images ───────────────────────────────
results = defaultdict(list)
latencies = []

for cls_dir in IMAGE_DIR.iterdir():
    if not cls_dir.is_dir():
        continue
    true_idx = class_to_index[cls_dir.name]
    logger.info(f"Starting processing for class '{cls_dir.name}'.")

    for img_path in tqdm(cls_dir.glob('*')):
        img_np = preprocess(img_path)
        t0 = time.perf_counter()
        preds = infer(img_np)
        latency = time.perf_counter() - t0
        latencies.append(latency)

        pred_idx, score = preds[0]
        results[true_idx].append((pred_idx, score))

    logger.info(
        f"Completed processing for class '{cls_dir.name}'. "
        f"Total images: {len(results[true_idx])}."
    )

# ─── 6. Compute metrics ────────────────────────────────────────
total, correct = 0, 0
confusion = Counter()

# ★ NEW: per-class counters
per_class_total   = Counter()   # 각 클래스를 몇 번 평가했는가
per_class_correct = Counter()   # 각 클래스가 맞은 횟수

for true_idx, lst in results.items():
    for pred_idx, _ in lst:
        confusion[(true_idx, pred_idx)] += 1
        per_class_total[true_idx]   += 1       # ★ NEW
        if pred_idx == true_idx:
            correct                 += 1
            per_class_correct[true_idx] += 1   # ★ NEW
        total += 1

acc = correct / total if total else 0
avg_latency = np.mean(latencies) * 1000

# ★ NEW: 클래스별 정확도 계산
class_acc = {
    index_to_class[idx]: per_class_correct[idx] / per_class_total[idx]
    for idx in per_class_total
}

# ─── 7. Report ────────────────────────────────────────────────
logger.info(f"Total image count: {total}")
logger.info(f"Top-1 accuracy: {acc:.4%}")
logger.info(f"Average inference time: {avg_latency:.2f} ms (Edge TPU)")

# ★ NEW: 클래스별 정확도 출력
logger.info("Per-class accuracy:")
for cls_name, a in sorted(class_acc.items()):
    logger.info(f"  {cls_name:<20}: {a:.4%}")

logger.info("Confusion matrix (entries ≥1):")
for (t, p), n in sorted(confusion.items()):
    if n:
        logger.info(f"{index_to_class[t]:<20} → {index_to_class[p]:<20}: {n}")
