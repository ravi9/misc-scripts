import time
import os
import shutil
import openvino
import openvino_genai as ov_genai

model_path = r"Qwen2.5-1.5B-Instruct-g128-int4-ov"
cache_base_dir = "cache-qwen-2.5"

blob_path = f"{cache_base_dir}/npu_cache_blob/compiled_model.blob"

# Delete the cache_base_dir folder if it exists
if os.path.exists(cache_base_dir):
    print(f"Deleting cache dir: {cache_base_dir}")
    shutil.rmtree(cache_base_dir)
    
dir_name = os.path.dirname(blob_path)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
    print(f"Created cache dir: {dir_name}")
    
def measure_performance(pipeline_config, config_name):
    print(f"\nMeasuring performance for: {config_name}")
    print(f"Pipe Config: {pipeline_config}")
    
    try:
        start_time = time.time()
        pipe = ov_genai.LLMPipeline(model_path, "NPU", **pipeline_config)
        load_time = time.time() - start_time
        print(f"{config_name}: Load time: {load_time:.2f} sec.")
    except Exception as e:
        print(f"{config_name}: Error initializing:\n {e}")
        return -1, -1
    
    # Inference
    try:
        start_time = time.time()
        result = pipe.generate("Sun is the largest ", max_new_tokens=50)
        infer_time = time.time() - start_time
        print(f"{config_name}: Infer time: {infer_time:.2f} sec.")
    except Exception as e:
        print(f"Error generating with pipeline for {config_name}: {e}")
        infer_time = -1
    
    del pipe
    
    return load_time, infer_time

print(f"\nOpenVINO: {openvino.__version__}")
print(f"ov_genai: {ov_genai.__version__}")

# No caching
load_time_no_cache, infer_time_no_cache = measure_performance({}, "No Cache")

# With NPUW_CACHE_DIR 1st run
pipeline_config_npuw = {"NPUW_CACHE_DIR": f"{cache_base_dir}/npuw_cache"}
load_time_npuw_1st, infer_time_npuw_1st = measure_performance(pipeline_config_npuw, "NPUW_CACHE_DIR 1st Run")

# With NPUW_CACHE_DIR 2nd Run
pipeline_config_npuw = {"NPUW_CACHE_DIR": f"{cache_base_dir}/npuw_cache"}
load_time_npuw_2nd, infer_time_npuw_2nd = measure_performance(pipeline_config_npuw, "NPUW_CACHE_DIR 2nd Run")

# With CACHE_DIR 1st run
pipeline_config_cache = {"CACHE_DIR": f"{cache_base_dir}/npu_cache_dir", "CACHE_MODE" : "OPTIMIZE_SPEED"}
load_time_cache_1st, infer_time_cache_1st = measure_performance(pipeline_config_cache, "CACHE_DIR 1st Run")

# With CACHE_DIR 2nd run
pipeline_config_cache = {"CACHE_DIR": f"{cache_base_dir}/npu_cache_dir", "CACHE_MODE" : "OPTIMIZE_SPEED"}
load_time_cache_2nd, infer_time_cache_2nd = measure_performance(pipeline_config_cache, "CACHE_DIR 2nd Run")

# With CACHE_DIR - NO CACHE_MODE 1st run
pipeline_config_cache = {"CACHE_DIR": f"{cache_base_dir}/npu_cache_dir_no_cache_mode", }
load_time_cache_nomode_1st, infer_time_cache_nomode_1st = measure_performance(pipeline_config_cache, "CACHE_DIR_NO_CACHE_MODE 1st Run")

# With CACHE_DIR - NO CACHE_MODE 2nd run
pipeline_config_cache = {"CACHE_DIR": f"{cache_base_dir}/npu_cache_dir_no_cache_mode"}
load_time_cache_nomode_2nd, infer_time_cache_nomode_2nd = measure_performance(pipeline_config_cache, "CACHE_DIR_NO_CACHE_MODE 2nd Run")

# With AOT compilation 1st run
pipeline_config_aot = {"EXPORT_BLOB": "YES", "BLOB_PATH": blob_path, "CACHE_MODE" : "OPTIMIZE_SPEED"}
load_time_aot_1st, infer_time_aot_1st = measure_performance(pipeline_config_aot, "AOT 1st Run")

# AOT compilation 2nd run
pipeline_config_aot_2 = {"BLOB_PATH": blob_path, "CACHE_MODE" : "OPTIMIZE_SPEED"}
load_time_aot_2nd, infer_time_aot_2nd = measure_performance(pipeline_config_aot_2, "AOT 2nd Run")

# Print results
print(f"Model Path: {model_path}")
print(f"Cache Dir: {cache_base_dir}")

print("\nLoad Times:")
print(f"No Cache: {load_time_no_cache:.2f} sec")
print(f"NPUW_CACHE_DIR 1st Run: {load_time_npuw_1st:.2f} sec")
print(f"NPUW_CACHE_DIR 2nd Run: {load_time_npuw_2nd:.2f} sec")
print(f"CACHE_DIR 1st Run: {load_time_cache_1st:.2f} sec")
print(f"CACHE_DIR 2nd Run: {load_time_cache_2nd:.2f} sec")
print(f"CACHE_DIR_NO_CACHE_MODE 1st Run: {load_time_cache_nomode_1st:.2f} sec")
print(f"CACHE_DIR_NO_CACHE_MODE 2nd Run: {load_time_cache_nomode_2nd:.2f} sec")
print(f"AOT First Run: {load_time_aot_1st:.2f} sec")
print(f"AOT Second Run: {load_time_aot_2nd:.2f} sec")

print("\nInference Times:")
print(f"No Cache: {infer_time_no_cache:.2f} sec")
print(f"NPUW_CACHE_DIR 1st Run: {infer_time_npuw_1st:.2f} sec")
print(f"NPUW_CACHE_DIR 2nd Run: {infer_time_npuw_2nd:.2f} sec")
print(f"CACHE_DIR 1st Run: {infer_time_cache_1st:.2f} sec")
print(f"CACHE_DIR 2nd Run: {infer_time_cache_2nd:.2f} sec")
print(f"CACHE_DIR_NO_CACHE_MODE 1st Run: {infer_time_cache_nomode_1st:.2f} sec")
print(f"CACHE_DIR_NO_CACHE_MODE 2nd Run: {infer_time_cache_nomode_2nd:.2f} sec")
print(f"AOT 1st Run: {infer_time_aot_1st:.2f} sec")
print(f"AOT 2nd Run: {infer_time_aot_2nd:.2f} sec")