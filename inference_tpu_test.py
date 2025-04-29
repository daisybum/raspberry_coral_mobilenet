from tflite_runtime.interpreter import Interpreter, load_delegate

model_path = "models/mobilenet_int8_edgetpu.tflite"

# 1. Create interpreter with Edge TPU Delegate
try:
    interpreter = Interpreter(model_path=model_path,
                              experimental_delegates=[load_delegate('libedgetpu.so.1')])
    interpreter.allocate_tensors()
    print("Successfully loaded Edge TPU delegate")
except Exception as e:
    print("Failed to load Edge TPU delegate:", e)
    # Fallback to CPU interpreter if delegate loading fails
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    print("Running on CPU without Edge TPU")

# 2. Check delegates information of the interpreter object
if hasattr(interpreter, "_delegates") and interpreter._delegates:
    # Print list of loaded delegates
    print("List of currently used Delegate objects:", interpreter._delegates)
    # Inspect each Delegate object and print detailed information
    for delegate in interpreter._delegates:
        # Extract the library name loaded by the delegate object (libedgetpu.so for Edge TPU)
        lib_name = getattr(getattr(delegate, "_library", None), "_name", None)
        if lib_name:
            print(f"Loaded Delegate library: {lib_name}")
        else:
            print("Edge TPU Delegate is loaded and in use")
else:
    print("No Delegate loaded (Running on CPU)")

# 3. Check for Edge TPU custom operations in the model
print(len(interpreter._get_ops_details()))
for op_index in range(len(interpreter._get_ops_details())):
    print(op_index, interpreter._get_op_details(op_index)['op_name'])
    if interpreter._get_op_details(op_index)['op_name'] == 'edgetpu-custom-op':
        print("✔️  edgetpu-custom-op 발견 → Edge TPU 가속 활성")
        break
