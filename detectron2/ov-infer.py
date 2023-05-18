from openvino.runtime import Core
from openvino.runtime import Layout, set_batch
import numpy as np
import time

ir_path="./rp-output/IR/model.xml"
ie = Core()
model = ie.read_model(ir_path)

# Update batch size
# model.get_parameters()[0].set_layout(Layout("N..."))
# set_batch(model,1)

# Update shape
# The default input shape is [C,H,W] [3,800,1202]
# The H can be one of these values (640, 672, 704, 736, 768, 800) Refer: https://github.com/facebookresearch/detectron2/blob/main/configs/Base-RCNN-FPN.yaml#L41

input_shape = [3, 736, 1200]
model.reshape(input_shape)

# Compile the model
compiled_model = ie.compile_model(model=model, device_name="CPU")

# get the names of input and output layers of the model
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

bench_time = 5 # benchmark time in seconds
dummy_input = np.random.randn(*tuple(input_shape))

latency_arr = []
end = time.time() + bench_time

print(f"\nBenchmarking OpenVINO inference for {bench_time}sec...")
print(f"Input shape: {dummy_input.shape}")
print(f"Model: {ir_path} ")

while time.time() < end:
    start_time = time.time()
    ov_result = compiled_model([dummy_input])
    latency = time.time() - start_time
    latency_arr.append(latency)

# Save the result for accuracy verificaiton
print(f"Output shapes:")
for out_nm in compiled_model.outputs:
    print(f"{ov_result[out_nm].shape}")

avg_latency = np.array(latency_arr).mean()
fps = 1 / avg_latency

print(f"\nAvg Latency: {avg_latency:.4f} sec, FPS: {fps:.2f}\n")

