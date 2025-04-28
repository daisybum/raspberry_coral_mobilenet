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
    model_path='models/mobilenet_int8_edgetpu.tflite',
    experimental_delegates=[delegate]
)
interpreter.allocate_tensors()
# 3) Check input/output information
print("Input details:", interpreter.get_input_details())
print("Output details:", interpreter.get_output_details())
