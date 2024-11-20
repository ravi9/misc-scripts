import time
from diffusers import UNet2DConditionModel, DiffusionPipeline, LCMScheduler
import torch
from PIL import Image
import openvino.torch
from typing import Dict, Tuple, List
import json
from datetime import datetime
from enum import Enum, auto
import numpy as np
import argparse
import sys
import importlib.metadata

class RunMode(Enum):
    EAGER = "eager"
    TC_INDUCTOR = "tc_inductor"
    TC_OPENVINO = "tc_openvino"

def setup_pipeline(run_mode: str, ckpt: str, dtype=torch.float16) -> DiffusionPipeline:
    """
    Setup the diffusion pipeline based on run mode configuration
    
    Args:
        run_mode: One of 'eager', 'tc_inductor', or 'tc_openvino'
        ckpt: Path to the model checkpoint
        dtype: Model dtype
    """
    print(f"\nInitializing pipeline with mode: {run_mode}")
    
    # Set compile options based on run mode
    if run_mode == RunMode.TC_OPENVINO.value:
        compile_options = {
            'backend': 'openvino',
            'options': {'device': 'CPU', 'config': {'PERFORMANCE_HINT': 'LATENCY'}}
        }
        print(f"Using OpenVINO backend with options: {compile_options}")
    elif run_mode == RunMode.TC_INDUCTOR.value:
        compile_options = {'backend': 'inductor', 'options': {}}
        print(f"Using Inductor backend with options: {compile_options}")
    else:  # eager mode
        compile_options = {}
        print("Using eager mode (no compilation)")
    
    # Initialize models
    unet = UNet2DConditionModel.from_pretrained(f"{ckpt}/lcm/", torch_dtype=dtype)
    pipe = DiffusionPipeline.from_pretrained(ckpt, unet=unet, torch_dtype=dtype)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    
    # Apply compilation if using torch.compile
    if run_mode != RunMode.EAGER.value:
        print("Compiling models...")
        pipe.text_encoder = torch.compile(pipe.text_encoder, **compile_options)
        pipe.unet = torch.compile(pipe.unet, **compile_options)
        pipe.vae.decode = torch.compile(pipe.vae.decode, **compile_options)
    
    pipe.to("cpu")
    return pipe

def run_inference(pipe: DiffusionPipeline, params: Dict, iteration: int = 0) -> Tuple[Image.Image, float]:
    """Run inference and measure time"""
    start_time = time.time()
    image = pipe(
        params["prompt"],
        num_inference_steps=params["num_inference_steps"],
        guidance_scale=params["guidance_scale"],
        height=params["height"],
        width=params["width"],
    ).images[0]
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"Iteration {iteration} execution time: {execution_time:.2f} seconds")
    
    return image, execution_time

def run_benchmark(run_mode: str, params: Dict, num_iter: int) -> Dict:
    """Run a single benchmark configuration with multiple iterations"""
    try:
        pipe = setup_pipeline(
            run_mode,
            params["ckpt"],
            params["dtype"]
        )
        
        # Warm-up run
        print("\nPerforming warm-up run...")
        warmup_image, warmup_time = run_inference(pipe, params, iteration=0)
        
        # Benchmark iterations
        print(f"\nRunning {num_iter} benchmark iterations...")
        iteration_times = []
        final_image = None
        
        for i in range(num_iter):
            image, exec_time = run_inference(pipe, params, iteration=i+1)
            iteration_times.append(exec_time)
            if i == num_iter - 1:
                final_image = image
        
        # Calculate statistics
        stats = {
            "mean": float(np.mean(iteration_times)),
            "median": float(np.median(iteration_times)),
            "std": float(np.std(iteration_times)),
            "min": float(np.min(iteration_times)),
            "max": float(np.max(iteration_times)),
            "percentile_90": float(np.percentile(iteration_times, 90)),
            "percentile_95": float(np.percentile(iteration_times, 95)),
            "all_iterations": iteration_times
        }
        
        # Save images
        warmup_image_filename = f"image-{run_mode}-warmup.png"
        final_image_filename = f"image-{run_mode}-final.png"
        warmup_image.save(warmup_image_filename)
        final_image.save(final_image_filename)
        
        return {
            "run_mode": run_mode,
            "warmup_time": warmup_time,
            "statistics": stats,
            "warmup_image": warmup_image_filename,
            "final_image": final_image_filename,
            "status": "success"
        }
    except Exception as e:
        print(f"Error during benchmark: {str(e)}")
        return {
            "run_mode": run_mode,
            "status": "failed",
            "error": str(e)
        }

def save_results(results: List[Dict], sw_versions: List[Dict], filename: str = None):
    """Save benchmark results to a JSON file"""
    if filename is None:
        filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
        json.dump(sw_versions, f, indent=2)
        
    print(f"\nResults saved to {filename}")

def get_sw_versions():
    sw_versions = {}
    packages = [
        ("Python", "python"),
        ("TorchServe", "torchserve"),
        ("OpenVINO", "openvino"),
        ("PyTorch", "torch"),
        ("Transformers", "transformers"),
        ("Diffusers", "diffusers")
    ]

    for name, package in packages:
        try:
            version = importlib.metadata.version(package)
            sw_versions[name] = version
        except Exception as e:
            sw_versions[name] = "Not installed"

    return sw_versions
    
def main():

    # Parse command-line args
    parser = argparse.ArgumentParser(description='Stable Diffusion Benchmark script')
    parser.add_argument('-ni', '--num_iter', type=int, default=3, help='Number of benchmark iterations')
    args = parser.parse_args()
    
    # Number of benchmark iterations
    num_iter = args.bench_iter

    # Run modes to test
    run_modes = [
        RunMode.EAGER.value,
        RunMode.TC_INDUCTOR.value,
        RunMode.TC_OPENVINO.value
    ]
    
    # Parameters
    params = {
        "ckpt": "/home/model-server/model-store/stabilityai---stable-diffusion-xl-base-1.0/model",
        "guidance_scale": 5.0,
        "num_inference_steps": 4,
        "height": 768,
        "width": 768,
        "prompt": "a close-up picture of an old man standing in the rain",
        "dtype": torch.float16
    }
    
    # Run benchmarks
    results = []
    for mode in run_modes:
        print("\n" + "="*50)
        print(f"Running benchmark with run mode: {mode}")
        print(f"Number of iterations: {num_iter}")
        print("="*50)
        
        result = run_benchmark(mode, params, num_iter)
        results.append(result)
    
    sw_versions = get_sw_versions()
    print("\nSoftware Versions:")
    print("-"*50)
    for name, version in sw_versions.items():
        print(f"{name}: {version}")

    # Save results
    save_results(results, sw_versions)
    
    # Print summary
    print("\nBenchmark Summary:")
    print("-"*50)
    for result in results:
        if result["status"] == "success":
            print(f"\nRun Mode: {result['run_mode']}")
            print(f"  Warm-up Time: {result['warmup_time']:.2f} seconds")
            print(f"  Average Time: {result['statistics']['mean']:.2f} seconds")
            print(f"  Median Time: {result['statistics']['median']:.2f} seconds")
            print(f"  Std Dev: {result['statistics']['std']:.2f} seconds")
            print(f"  Min Time: {result['statistics']['min']:.2f} seconds")
            print(f"  Max Time: {result['statistics']['max']:.2f} seconds")
            print(f"  90th Percentile: {result['statistics']['percentile_90']:.2f} seconds")
            print(f"  95th Percentile: {result['statistics']['percentile_95']:.2f} seconds")
            print(f"  Warm-up image saved as: {result['warmup_image']}")
            print(f"  Final image saved as: {result['final_image']}")
        else:
            print(f"\nRun Mode: {result['run_mode']}")
            print(f"  Status: Failed")
            print(f"  Error: {result['error']}")

if __name__ == "__main__":
    main()

# Usage: python torchcompile-sdxl-lcm-benchmark.py -ni 3
