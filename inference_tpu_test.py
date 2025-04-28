#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Edge TPU MobileNetV3-Large Inference Script (with Logger)
"""
import os, time, json, itertools, logging
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
from pycoral.utils import edgetpu
from pycoral.adapters import common, classify
from tflite_runtime.interpreter import Interpreter, load_delegate
import os, time
# Enable internal logging
os.environ['EDGETPU_LOG'] = '1'
# 1) Load Delegate
delegate = load_delegate('libedgetpu.so.1')
print("Delegate loaded successfully:", delegate)
# 2) Create Interpreter
interpreter = Interpreter(
    model_path='models/your_model_edgetpu.tflite',
    experimental_delegates=[delegate]
)
interpreter.allocate_tensors()
# 3) Check input/output information
print("Input details:", interpreter.get_input_details())
print("Output details:", interpreter.get_output_details())
# ─── 0. Path Configuration ───────────────────────────────────────────────
IMAGE_DIR = Path('/workspace/testset')
MODEL_PATH = Path('models/mobilenet_int8_edgetpu.tflite')
TOP_K = 1
logger.info("Path configuration completed.")
# ─── 1. Initialize Edge TPU Interpreter ───────────────────────────────
interpreter = edgetpu.make_interpreter(str(MODEL_PATH), device="usb")
interpreter.allocate_tensors()
input_width, input_height = common.input_size(interpreter)
logger.info(f"Model input size: {input_width}×{input_height} RGB")
# ─── 2. Extract Class Labels (Based on Folder Names) ────────────────────────────
class_to_index = {cls: idx for idx, cls in
                  enumerate(sorted([d.name for d in IMAGE_DIR.iterdir() if d.is_dir()]))}
index_to_class = {v: k for k, v in class_to_index.items()}
logger.info(f"Class label extraction completed: {json.dumps(index_to_class, ensure_ascii=False)}")
# ─── 3. Helper Functions: Preprocessing & Inference ────────────────────────────────
def preprocess(img_path):
    img = Image.open(img_path).convert('RGB').resize((input_width, input_height), Image.BILINEAR)
    return np.asarray(img).astype(np.uint8)
def infer(image_np):
    common.set_input(interpreter, image_np)
    interpreter.invoke()
    classes = classify.get_classes(interpreter, top_k=TOP_K)
    return [(c.id, c.score) for c in classes]
# ─── 4. Process All Images ────────────────────────────────────────
results = defaultdict(list)
latencies = []
for cls_dir in IMAGE_DIR.iterdir():
    if not cls_dir.is_dir():
        continue
    true_idx = class_to_index[cls_dir.name]
    logger.info(f"Starting to process images for class '{cls_dir.name}'.")
    for img_path in tqdm(cls_dir.glob('*')):
        img_np = preprocess(img_path)
        t0 = time.perf_counter()
        preds = infer(img_np)
        latency = time.perf_counter() - t0
        latencies.append(latency)
        pred_idx, score = preds[0]
        results[true_idx].append((pred_idx, score))
        print("Sleep 1 seconds from now on...")
        time.sleep(1)
        print("wake up!")
    logger.info(f"Processing completed for class '{cls_dir.name}'. Total {len(results[true_idx])} images processed.")
# ─── 5. Calculate Metrics ────────────────────────────────────────────────
total, correct = 0, 0
confusion = Counter()
for true_idx, lst in results.items():
    for pred_idx, _ in lst:
        confusion[(true_idx, pred_idx)] += 1
        total += 1
        if pred_idx == true_idx:
            correct += 1
acc = correct / total if total else 0
avg_latency = np.mean(latencies) * 1000
# ─── 6. Output Report ─────────────────────────────────────────────
logger.info(f"Total number of images: {total}")
logger.info(f"Top-1 Accuracy: {acc:.4%}")
logger.info(f"Average inference time: {avg_latency:.2f} ms (Edge TPU)")
logger.info("Confusion matrix (items with sample count ≥1):")
for (t, p), n in sorted(confusion.items()):
    if n > 0:
        logger.info(f"{index_to_class[t]:<20} → {index_to_class[p]:<20}: {n}")
