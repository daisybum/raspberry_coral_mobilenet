#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Edge TPU MobileNetV3-Large 추론 스크립트
* testset/
   ├─ hazy/
   │   ├─ img001.jpg …
   ├─ normal/
   └─ …
"""

import os, time, json, itertools
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict, Counter

from pycoral.utils import edgetpu
from pycoral.adapters import common, classify  # ⬅️ pycoral API

# ─── 0. 경로 설정 ───────────────────────────────────────────────
IMAGE_DIR   = Path('/workspace/testset')      # 클래스별 하위 폴더
MODEL_PATH  = Path('models/mobilenet_int8.tflite')
TOP_K       = 1                               # Top-k 결과 중 하나만 사용

# ─── 1. Edge TPU 인터프리터 초기화 ───────────────────────────────
interpreter = edgetpu.make_interpreter(str(MODEL_PATH))
interpreter.allocate_tensors()

input_width, input_height = common.input_size(interpreter)
print(f"⚙️  Model expects {input_width}×{input_height} RGB")

# ─── 2. 클래스 라벨 추출(폴더명 기준) ────────────────────────────
class_to_index = {cls: idx for idx, cls in
                  enumerate(sorted([d.name for d in IMAGE_DIR.iterdir() if d.is_dir()]))}
index_to_class = {v: k for k, v in class_to_index.items()}
print(json.dumps(index_to_class, indent=2, ensure_ascii=False))

# ─── 3. 보조 함수: 전처리 & 추론 ────────────────────────────────
def preprocess(img_path):
    """PIL.Image → 모델 입력 텐서(float32 또는 uint8)"""
    img = Image.open(img_path).convert('RGB').resize(
        (input_width, input_height), Image.BILINEAR)
    arr = np.asarray(img).astype(np.uint8)         # 모델이 uint8 입·출력
    return arr

def infer(image_np):
    """Numpy(HxWx3 uint8) → Top-k 클래스 인덱스"""
    common.set_input(interpreter, image_np)
    interpreter.invoke()
    classes = classify.get_classes(interpreter, top_k=TOP_K)  # score 포함 객체 리스트
    return [(c.id, c.score) for c in classes]

# ─── 4. 전체 이미지 순회 ────────────────────────────────────────
results = defaultdict(list)   # {true_cls: [(pred_cls, score), …]}
latencies = []

for cls_dir in IMAGE_DIR.iterdir():
    if not cls_dir.is_dir():          # 파일 스킵
        continue
    true_idx = class_to_index[cls_dir.name]
    for img_path in cls_dir.glob('*'):
        img_np = preprocess(img_path)
        t0 = time.perf_counter()
        preds = infer(img_np)
        latencies.append(time.perf_counter() - t0)

        pred_idx, score = preds[0]
        results[true_idx].append((pred_idx, score))

# ─── 5. 지표 계산 ────────────────────────────────────────────────
total, correct = 0, 0
confusion = Counter()  # (true, pred) 쌍 카운트

for true_idx, lst in results.items():
    for pred_idx, _ in lst:
        confusion[(true_idx, pred_idx)] += 1
        total += 1
        if pred_idx == true_idx:
            correct += 1

acc = correct / total if total else 0
avg_latency = np.mean(latencies) * 1000  # ms

# ─── 6. 리포트 출력 ─────────────────────────────────────────────
print(f"\n📊 전체 이미지: {total}")
print(f"✅ Top-1 정확도 : {acc:.4%}")
print(f"⏱  평균 추론 시간: {avg_latency:.2f} ms (Edge TPU)")

print("\n🔍 혼동 행렬(샘플 수 ≥1인 항목만):")
for (t, p), n in sorted(confusion.items()):
    if n > 0:
        print(f"  {index_to_class[t]:<20} → {index_to_class[p]:<20}: {n}")
