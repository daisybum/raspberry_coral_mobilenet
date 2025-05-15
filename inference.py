#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, json, logging
from pathlib import Path
import numpy as np
from PIL import Image

from pycoral.utils import edgetpu
from pycoral.adapters import common, classify

# ─── 0. Logger ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# ─── 1. Edge-TPU 확인 및 인터프리터 초기화 ────────────────────
devices = edgetpu.list_edge_tpus()
using_tpu = bool(devices)

MODEL_PATH = Path('models/model_quant_edgetpu.tflite')
if using_tpu:
    log.info(f"✅ Edge TPU detected: {devices}")
    interpreter = edgetpu.make_interpreter(str(MODEL_PATH), device="usb")
else:
    log.warning("⚠️ Edge TPU not found – CPU fallback")
    from tflite_runtime.interpreter import Interpreter
    interpreter = Interpreter(model_path=str(MODEL_PATH))

interpreter.allocate_tensors()
input_w, input_h = common.input_size(interpreter)
log.info(f"Model input size: {input_w}×{input_h} RGB")

# ─── 2. 클래스 라벨 로드 (labels.txt 사용 또는 직접 지정) ───
LABELS_PATH = Path('models/labels.txt')
if LABELS_PATH.exists():
    labels = LABELS_PATH.read_text().splitlines()
else:
    labels = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4']
log.info(f"Labels: {json.dumps(labels, ensure_ascii=False)}")

# ─── 3. 보조 함수 ─────────────────────────────────────────────
def preprocess(img_path: Path) -> np.ndarray:
    """PIL → Numpy (uint8) + 리사이즈"""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((input_w, input_h), Image.BILINEAR)
    return np.asarray(img).astype(np.uint8)

def infer(img_np: np.ndarray):
    """단일 이미지 추론 → (pred_id, score)"""
    common.set_input(interpreter, img_np)
    t0 = time.perf_counter()
    interpreter.invoke()
    latency = (time.perf_counter() - t0) * 1000  # ms
    preds = classify.get_classes(interpreter, top_k=1)
    pred_id, score = preds[0].id, preds[0].score
    return pred_id, score, latency

# ─── 4. 캡처 & 추론 루프 ─────────────────────────────────────
CAPTURE_DIR = Path('./captured_images')
CAPTURE_DIR.mkdir(exist_ok=True, parents=True)
CAPTURE_CMD = "libcamera-still -n -o {dst} --width 1640 --height 1232"
INTERVAL = 30  # seconds

log.info("=== Start realtime inference loop ===")
while True:
    ts = time.strftime("%Y%m%d_%H%M%S")
    img_file = CAPTURE_DIR / f"capture_{ts}.jpg"
    cmd = CAPTURE_CMD.format(dst=img_file)
    log.info(f"📷 Capturing… ({cmd})")
    os.system(cmd)

    if not img_file.exists():
        log.warning(f"Capture failed: {img_file} not found")
        time.sleep(INTERVAL)
        continue

    img_np = preprocess(img_file)
    pred_id, score, latency = infer(img_np)
    pred_label = labels[pred_id] if pred_id < len(labels) else f"id_{pred_id}"

    log.info(f"{img_file.name} → {pred_label} (score={score:.3f}, {latency:.1f} ms)")

    # 저장 공간 절약: 추론 후 이미지 삭제하려면 주석 해제
    img_file.unlink(missing_ok=True)

    log.info(f"Waiting {INTERVAL}s...\n")
    time.sleep(INTERVAL)
