#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Edge TPU MobileNetV3-Large Inference Script (with Logger)
"""
import warnings
import os, time, json, itertools, logging
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
from pycoral.utils import edgetpu
from pycoral.adapters import common, classify
from tflite_runtime.interpreter import Interpreter, load_delegate

# DeprecationWarning 무시
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Enable internal logging
os.environ['EDGETPU_LOG'] = '1'

# 1) Load Delegate
delegate = load_delegate('libedgetpu.so.1')

# 2) Create Interpreter
interpreter = Interpreter(
    model_path='models/mobilenet_int8_edgetpu.tflite',
    experimental_delegates=[delegate]
)
interpreter.allocate_tensors()

# 3) Check input/output information
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def safe_print(detail_list, title):
    print(f"\n=== {title} ===")
    for idx, detail in enumerate(detail_list):
        print(f"\n-- #{idx} --")
        for k, v in detail.items():
            # ndarray든 그 외 객체든 tolist() 가능하면 호출
            if hasattr(v, "tolist"):
                printable = v.tolist()
            else:
                printable = v
            print(f"{k}: {printable}")

# 실제 출력
safe_print(input_details,  "Input details")
safe_print(output_details, "Output details")
