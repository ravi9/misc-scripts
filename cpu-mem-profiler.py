'''
Example Usage:
 
python cpu-mem-profiler.py -p "python my-script.py -n 'myarg' "

To print peak memory usage in GB:
python cpu-mem-profiler.py -p "python my-script.py -n 'myarg' " -m GB

memory_info().rss: returns the resident set size, which is the non-swapped physical memory a process has used (in bytes).
'''

import psutil
import subprocess
import time
import argparse
import shlex

def monitor_program(program, memory_unit='MB'):
    # Split the program string into arguments
    program_args = shlex.split(program)

    # Start the program
    process = subprocess.Popen(program_args)

    # Initialize peak usage variables
    peak_cpu = 0
    peak_memory = 0

    memory_factor = 1  # Default is MB
    if memory_unit == 'GB':
        memory_factor = 1024  # Convert MB to GB

    try:
        while True:
            # Get process info
            p = psutil.Process(process.pid)

            # Update peak CPU and memory usage
            cpu_usage = p.cpu_percent(interval=1) / psutil.cpu_count()
            memory_usage = p.memory_info().rss / (1024 * 1024 * memory_factor)  # Convert to MB or GB

            if cpu_usage > peak_cpu:
                peak_cpu = cpu_usage

            if memory_usage > peak_memory:
                peak_memory = memory_usage

            # Check if the process has terminated
            if process.poll() is not None:
                break
    except psutil.NoSuchProcess:
        pass

    print(f"Finished profiling: {program}")
    # Output the peak usage
    print(f"Peak CPU Usage: {peak_cpu:.2f}%")
    print(f"Peak Memory Usage: {peak_memory:.2f} {memory_unit}")
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monitor peak CPU and memory usage of a program.')
    parser.add_argument('-p', '--program', required=True, help='The program to run and monitor, including any arguments, enclosed in quotes.')
    parser.add_argument('-m', '--memory-unit', choices=['MB', 'GB'], default='MB', help='The unit for memory usage (MB or GB). Default is MB.')

    args = parser.parse_args()

    monitor_program(args.program, args.memory_unit)
