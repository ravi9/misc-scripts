"""
OpenVINO NPU Caching Performance Benchmark Tool

"""

import time
import os
import shutil
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import openvino as ov
import openvino_genai as ov_genai
import argparse


@dataclass
class BenchmarkResult:
    """Stores benchmark results for a single configuration."""
    config_name: str
    compile_time: float
    inference_time: float
    success: bool = True
    error_message: Optional[str] = None


class NpuCachePerfTool:
    """Benchmarks OpenVINO LLM pipeline performance with different caching strategies."""
    
    def __init__(self, model_path: str, cache_base_dir: str, device: str = "NPU"):
        self.model_path = model_path
        self.cache_base_dir = cache_base_dir
        self.device = device
        self.blob_path = f"{cache_base_dir}/npu_cache_aot/compiled_model.blob"
        self.results = []
        
        self._setup_cache_directory()
        self._print_environment_info()
    
    def _setup_cache_directory(self) -> None:
        """Clean up and recreate cache directories."""
        if os.path.exists(self.cache_base_dir):
            print(f"\n!!! Deleting existing cache directory: {self.cache_base_dir} ... !!!\n")
            shutil.rmtree(self.cache_base_dir)
        
        # Create blob directory structure
        blob_dir = os.path.dirname(self.blob_path)
        if not os.path.exists(blob_dir):
            os.makedirs(blob_dir)
            print(f"Created cache directory: {blob_dir}")
    
    def _print_environment_info(self) -> None:
        """Print version information."""
        ov_core = ov.Core()
    
        print(f"OpenVINO version: {ov.__version__}")
        print(f"OpenVINO GenAI version: {ov_genai.__version__}")
        print(f"CPU Name: {ov_core.get_property("CPU", "FULL_DEVICE_NAME")}")
        print(f"NPU Driver: {ov_core.get_property('NPU', 'NPU_DRIVER_VERSION')}")
        print(f"NPU_MAX_TILES: {ov_core.get_property('NPU', 'NPU_MAX_TILES')}")
        print(f"NPU_DEVICE_TOTAL_MEM_SIZE: {ov_core.get_property('NPU', 'NPU_DEVICE_TOTAL_MEM_SIZE') / (1024 ** 3):.2f} GB")
        print(f"Model path: {self.model_path}")
        print(f"Cache base directory: {self.cache_base_dir}")
        print("-" * 60)
    
    def _get_directory_info(self, directory_path: str) -> Tuple[float, int]:
        """Calculate the total size of a directory in MB and count files."""
        if not os.path.exists(directory_path):
            return 0.0, 0

        total_size = 0
        file_count = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory_path):
                file_count += len(filenames)
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)
        except (OSError, IOError):
            return 0.0, -1  

        size_mb = total_size / (1024 * 1024)  # Convert to MB
        return size_mb, file_count
    
    
    def _measure_single_config(self, pipeline_config: Dict[str, Any], config_name: str) -> BenchmarkResult:
        """
        Measure performance for a single pipeline configuration.
        
        Args:
            pipeline_config: Configuration parameters for the pipeline
            config_name: Human-readable name for this configuration
            
        Returns:
            BenchmarkResult containing timing data and success status
        """
        print(f"\nBenchmarking: {config_name}")
        print(f"Configuration: {pipeline_config}")
        
        # Measure load time
        try:
            start_time = time.time()
            pipe = ov_genai.LLMPipeline(self.model_path, self.device, **pipeline_config)
            compile_time = time.time() - start_time
            print(f"✓ Compile time: {compile_time:.2f} seconds")
        except Exception as e:
            error_msg = f"Failed to initialize pipeline: {e}"
            print(f"✗ {error_msg}")
            return BenchmarkResult(config_name, -1, -1, False, error_msg)
        
        # Measure inference time
        try:
            prompt = "Sun is the largest "
            max_tokens = 50
            
            start_time = time.time()
            result = pipe.generate(prompt, max_new_tokens=max_tokens)
            inference_time = time.time() - start_time
            print(f"✓ Inference time: {inference_time:.2f} seconds")
        except Exception as e:
            error_msg = f"Failed to generate text: {e}"
            print(f"✗ {error_msg}")
            return BenchmarkResult(config_name, compile_time, -1, False, error_msg)
        finally:
            # Clean up pipeline object
            if 'pipe' in locals():
                del pipe

        
        return BenchmarkResult(config_name, compile_time, inference_time)
    
    def _get_benchmark_configurations(self) -> Dict[str, Dict[str, Any]]:
        """
        Define all benchmark configurations to test.
        
        Returns:
            Dictionary mapping configuration names to their parameters
        """
        return {
            "No Cache": {},
            
            "NPUW_CACHE_DIR (1st Run)": {
                "NPUW_CACHE_DIR": f"{self.cache_base_dir}/npuw_cache_dir"
            },
            
            "NPUW_CACHE_DIR (2nd Run)": {
                "NPUW_CACHE_DIR": f"{self.cache_base_dir}/npuw_cache_dir"
            },
            
            "CACHE_DIR (1st Run)": {
                "CACHE_DIR": f"{self.cache_base_dir}/npu_cache_dir"
            },
            
            "CACHE_DIR (2nd Run)": {
                "CACHE_DIR": f"{self.cache_base_dir}/npu_cache_dir"
            },

            "CACHE_DIR OPT SIZE (1st Run)": {
                "CACHE_DIR": f"{self.cache_base_dir}/npu_cache_dir_opt_size",
                "CACHE_MODE": "OPTIMIZE_SIZE"
            },

            "CACHE_DIR OPT SIZE (2nd Run)": {
                "CACHE_DIR": f"{self.cache_base_dir}/npu_cache_dir_opt_size",
                "CACHE_MODE": "OPTIMIZE_SIZE"
            },

            "CACHE_DIR OPT SPEED (1st Run)": {
                "CACHE_DIR": f"{self.cache_base_dir}/npu_cache_dir_opt_speed",
                "CACHE_MODE": "OPTIMIZE_SPEED"
            },

            "CACHE_DIR OPT SPEED (2nd Run)": {
                "CACHE_DIR": f"{self.cache_base_dir}/npu_cache_dir_opt_speed",
                "CACHE_MODE": "OPTIMIZE_SPEED"
            },
            
            "AOT Compilation (1st Run)": {
                "EXPORT_BLOB": "YES",
                "BLOB_PATH": self.blob_path,
                "CACHE_MODE": "OPTIMIZE_SPEED"
            },
            
            "AOT Compilation (2nd Run)": {
                "BLOB_PATH": self.blob_path,
                "CACHE_MODE": "OPTIMIZE_SPEED"
            }
        }
    
    def run_benchmark(self) -> None:
        """Run the complete benchmark suite."""
        print("Starting OpenVINO LLM Pipeline Benchmark")
        print("=" * 60)
        
        configurations = self._get_benchmark_configurations()
        
        for config_name, config_params in configurations.items():
            result = self._measure_single_config(config_params, config_name)
            self.results.append(result)
            
    def print_summary(self) -> None:
        """Print a comprehensive summary of all benchmark results."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        if not self.results:
            print("No results to display.")
            return

        # Assume "No Cache" is always the first result
        no_cache_result = self.results[0] if self.results[0].config_name == "No Cache" else None

        print("\nLOAD/COMPILE TIMES:")
        print("-" * 75)

        prev_result = None
        for i, result in enumerate(self.results):
            line = f"{result.config_name:<35}: "
            if result.success and result.compile_time >= 0:
                line += f"{result.compile_time:>8.2f} sec"
                parts = []

                # Skip speedup for "No Cache" (i == 0)
                if i > 0:
                    if no_cache_result and no_cache_result.success and no_cache_result.compile_time > 0:
                        speedup_vs_no_cache = no_cache_result.compile_time / result.compile_time
                        parts.append(f"{speedup_vs_no_cache:.1f}x vs No cache")

                    # For 2nd Run: also show speedup vs 1st Run
                    if "(2nd Run)" in result.config_name and prev_result and prev_result.success and prev_result.compile_time > 0:
                        speedup_vs_1st = prev_result.compile_time / result.compile_time
                        parts.append(f"{speedup_vs_1st:.1f}x vs 1st run")

                    if parts:
                        line += " (" + "; ".join(parts) + ")"
                    else:
                        line += f"{'FAILED':>8}"
            print(line)
            print() if "(1st Run)" not in result.config_name else None
            prev_result = result

        print("\nINFERENCE TIMES:")
        print("-" * 75)

        prev_result = None
        for i, result in enumerate(self.results):
            line = f"{result.config_name:<35}: "
            if result.success and result.inference_time >= 0:
                line += f"{result.inference_time:>8.2f} sec"
                parts = []

                # Skip speedup for "No Cache" (i == 0)
                if i > 0:
                    if no_cache_result and no_cache_result.success and no_cache_result.inference_time > 0:
                        speedup_vs_no_cache = no_cache_result.inference_time / result.inference_time
                        parts.append(f"{speedup_vs_no_cache:.1f}x vs No cache")

                    # For 2nd Run: show speedup vs 1st Run
                    if "(2nd Run)" in result.config_name and prev_result and prev_result.success and prev_result.inference_time > 0:
                        speedup_vs_1st = prev_result.inference_time / result.inference_time
                        parts.append(f"{speedup_vs_1st:.1f}x vs 1st run")

                    if parts:
                        line += " (" + "; ".join(parts) + ")"
                    else:
                        line += f"{'FAILED':>8}"
            print(line)
            print() if "(1st Run)" not in result.config_name else None

            prev_result = result

        # Print any errors
        failed_results = [r for r in self.results if not r.success]
        if failed_results:
            print("\nERRORS:")
            print("-" * 40)
            for result in failed_results:
                print(f"{result.config_name}: {result.error_message}")
            
    def print_cache_dir_info(self) -> None:
        """Print all cache directories and their sizes and file counts."""
        print("\n" + "=" * 60)
        print("CACHE DIRECTORY INFO ")
        print("=" * 60)
        
        if not os.path.exists(self.cache_base_dir):
            print("No cache base directory found.")
            return
        
        print(f"\nDirectories in '{self.cache_base_dir}':")
        print("-" * 60)
        print(f"{'Directory':<35} {'Size (MB)':>10} {'Files':>8}")
        print("-" * 60)
        
        # Loop through all directories in cache_base_dir
        for item in sorted(os.listdir(self.cache_base_dir)):
            item_path = os.path.join(self.cache_base_dir, item)
            if os.path.isdir(item_path):
                size_mb, file_count = self._get_directory_info(item_path)

                if file_count == -1:
                    print(f"{item:<35} {size_mb:>10.2f} MB {'ERROR':>8}")
                else:
                    print(f"{item:<35} {size_mb:>10.2f} MB {file_count:>8}")


def main():
    # Configuration
    parser = argparse.ArgumentParser(description="OpenVINO NPU Cache Performance Benchmark Tool")
    parser.add_argument("-m", "--model-path", type=str, default="llama-3.2-1b-instruct-npu-ov",
                        help="Path or name of the model to benchmark")
    parser.add_argument("-c", "--cache-base-dir", type=str, default="ov-npu-cache",
                        help="Base directory for cache storage")
    args = parser.parse_args()

    model_path = args.model_path
    cache_base_dir = args.cache_base_dir
    
    # Run benchmark
    start = time.time()
    npu_cache_perf_tool = NpuCachePerfTool(model_path, cache_base_dir)
    npu_cache_perf_tool.run_benchmark()
    npu_cache_perf_tool.print_summary()
    npu_cache_perf_tool.print_cache_dir_info()
    print(f"\nTotal time taken: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()