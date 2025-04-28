import os, time, logging
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# …(중략)…

# 1) Delegate 로드
delegate = load_delegate('libedgetpu.so.1')

# 2) Interpreter 생성
interpreter = Interpreter(
    model_path='models/mobilenet_int8_edgetpu.tflite',
    experimental_delegates=[delegate]
)
interpreter.allocate_tensors()

# 3) input/output details 받아오기
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 4) numpy 배열 -> 리스트 변환 후 출력
for idx, detail in enumerate(input_details):
    print(f"=== Input detail #{idx} ===")
    for k, v in detail.items():
        if isinstance(v, np.ndarray):
            print(f"{k}: {v.tolist()}")
        else:
            print(f"{k}: {v}")
    print()

for idx, detail in enumerate(output_details):
    print(f"=== Output detail #{idx} ===")
    for k, v in detail.items():
        if isinstance(v, np.ndarray):
            print(f"{k}: {v.tolist()}")
        else:
            print(f"{k}: {v}")
    print()
