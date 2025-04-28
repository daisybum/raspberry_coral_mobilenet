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

# 로거 초기화
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ─── 0. Edge TPU 탐지 ────────────────────────────────────────
devices = edgetpu.list_edge_tpus()
using_tpu = bool(devices)

if using_tpu:
    logger.info(f"✅ Edge TPU 장치 발견: {devices} – TPU에서 실행합니다.")
else:
    logger.warning("⚠️ Edge TPU 미발견 – CPU 모드로 실행합니다.")

# ─── 1. 인터프리터 초기화 ───────────────────────────────────────
if using_tpu:
    interpreter = edgetpu.make_interpreter(str(MODEL_PATH), device="usb")
else:
    from tflite_runtime.interpreter import Interpreter
    interpreter = Interpreter(model_path=str(MODEL_PATH))

interpreter.allocate_tensors()

# Delegate 로드 확인 (tflite_runtime에서는 private 속성일 수 있음)
loaded_delegates = getattr(interpreter, "_delegates", None)
if loaded_delegates:
    logger.info(f"로딩된 Delegate: {loaded_delegates}")
else:
    logger.info("로딩된 Delegate 없음 (CPU 모드)")

input_width, input_height = common.input_size(interpreter)
logger.info(f"모델 입력 크기: {input_width}×{input_height} RGB")

# ─── 2. 클래스 라벨 추출(폴더명 기준) ────────────────────────────
class_to_index = {cls: idx for idx, cls in
                  enumerate(sorted([d.name for d in IMAGE_DIR.iterdir() if d.is_dir()]))}
index_to_class = {v: k for k, v in class_to_index.items()}

logger.info(f"클래스 라벨 추출 완료: {json.dumps(index_to_class, ensure_ascii=False)}")

# ─── 3. 보조 함수: 전처리 & 추론 ────────────────────────────────
def preprocess(img_path):
    img = Image.open(img_path).convert('RGB').resize((input_width, input_height), Image.BILINEAR)
    return np.asarray(img).astype(np.uint8)

def infer(image_np):
    common.set_input(interpreter, image_np)
    interpreter.invoke()
    classes = classify.get_classes(interpreter, top_k=TOP_K)
    return [(c.id, c.score) for c in classes]

# ─── 4. 전체 이미지 순회 ────────────────────────────────────────
results = defaultdict(list)
latencies = []

for cls_dir in IMAGE_DIR.iterdir():
    if not cls_dir.is_dir():
        continue
    true_idx = class_to_index[cls_dir.name]
    logger.info(f"클래스 '{cls_dir.name}' 이미지 처리 시작.")

    for img_path in tqdm(cls_dir.glob('*')):
        img_np = preprocess(img_path)
        t0 = time.perf_counter()
        preds = infer(img_np)
        latency = time.perf_counter() - t0
        latencies.append(latency)

        pred_idx, score = preds[0]
        results[true_idx].append((pred_idx, score))

    logger.info(f"클래스 '{cls_dir.name}' 이미지 처리 완료. 총 {len(results[true_idx])}장 처리.")

# ─── 5. 지표 계산 ────────────────────────────────────────────────
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

# ─── 6. 리포트 출력 ─────────────────────────────────────────────
logger.info(f"전체 이미지 수: {total}")
logger.info(f"Top-1 정확도 : {acc:.4%}")
logger.info(f"평균 추론 시간: {avg_latency:.2f} ms (Edge TPU)")

logger.info("혼동 행렬(샘플 수 ≥1인 항목):")
for (t, p), n in sorted(confusion.items()):
    if n > 0:
        logger.info(f"{index_to_class[t]:<20} → {index_to_class[p]:<20}: {n}")
