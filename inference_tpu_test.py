from tflite_runtime.interpreter import Interpreter, load_delegate

model_path = "models/mobilenet_int8_edgetpu.tflite"

# 1. Edge TPU Delegate를 적용하여 인터프리터 생성
try:
    interpreter = Interpreter(model_path=model_path,
                              experimental_delegates=[load_delegate('libedgetpu.so.1')])
    interpreter.allocate_tensors()
    print("Edge TPU delegate를 성공적으로 로드했습니다.")
except Exception as e:
    print("Edge TPU delegate 로드 실패:", e)
    # delegate 로딩 실패 시 CPU 용 인터프리터로 폴백
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    print("Edge TPU 없이 CPU로 실행합니다.")

# 2. 인터프리터 객체의 delegates 정보 확인
if hasattr(interpreter, "_delegates") and interpreter._delegates:
    # 로드된 delegate 목록 출력
    print("현재 사용 중인 Delegate 객체 리스트:", interpreter._delegates)
    # Delegate 객체를 하나씩 조사하여 상세 정보 출력
    for delegate in interpreter._delegates:
        # delegate 객체가 로드한 라이브러리 이름 추출 (Edge TPU의 경우 libedgetpu.so)
        lib_name = getattr(getattr(delegate, "_library", None), "_name", None)
        if lib_name:
            print(f"로드된 Delegate 라이브러리: {lib_name}")
        else:
            print("Edge TPU Delegate가 로드되었으며 사용 중입니다.")
else:
    print("Delegate가 로드되지 않았습니다. (CPU 실행 중)")

# 3. 모델 내 Edge TPU 커스텀 연산 존재 여부 확인
try:
    op_index = 0
    while True:
        op_details = interpreter._get_op_details(op_index)
        op_name = op_details['op_name']
        if op_name == 'edgetpu-custom-op':
            print("모델에 'edgetpu-custom-op'이 포함되어 있어 Edge TPU 가속이 활성화됩니다.")
            break
        op_index += 1
except ValueError:
    # 더 이상 ops가 없음
    pass
