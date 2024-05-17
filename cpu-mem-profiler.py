'''
Example Usage:
 
python cpu-mem-profiler.py -p "python my-script.py -n 'myarg' "

To print memory usage in GB and interval of 0.5 seconds:
python cpu-mem-profiler.py -p "python my-script.py -n 'myarg' " -m GB -i 0.5

memory_info().rss: returns the resident set size, which is the non-swapped physical memory a process has used (in bytes).
'''

import psutil
import subprocess
import time
import argparse
import shlex

def monitor_program(program, memory_unit='MB', interval=1):
    # Split the program string into arguments
    program_args = shlex.split(program)

    # Start the program
    process = subprocess.Popen(program_args)

    peak_cpu = 0
    peak_memory = 0
    cpu_usage_list = []
    memory_factor = 1  # Default is MB
    if memory_unit == 'GB':
        memory_factor = 1024  # Convert MB to GB

    start_time = time.time()
    try:
        while True:
            # Get process info
            p = psutil.Process(process.pid)

            # Update peak CPU and memory usage
            cpu_usage = p.cpu_percent(interval=interval) / psutil.cpu_count()
            memory_usage = p.memory_info().rss / (1024 * 1024 * memory_factor)  # Convert to MB or GB

            cpu_usage_list.append(cpu_usage)
            
            if cpu_usage > peak_cpu:
                peak_cpu = cpu_usage

            if memory_usage > peak_memory:
                peak_memory = memory_usage

            # Check if the process has terminated
            if process.poll() is not None:
                break
    except psutil.NoSuchProcess:
        pass

    end_time = time.time()
    total_duration = end_time - start_time
    average_cpu = sum(cpu_usage_list) / len(cpu_usage_list) if cpu_usage_list else 0

    # Output the peak usage and other statistics
    print(f"\nFinished profiling: {program}")
    print(f"Profiling interval: {interval} sec")
    print(f"Total Duration: {total_duration:.2f} sec")
    print(f"Peak CPU Usage: {peak_cpu:.2f}%")
    print(f"Average CPU Usage: {average_cpu:.2f}%")
    print(f"Peak Memory Usage: {peak_memory:.2f} {memory_unit}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monitor peak CPU and memory usage of a program.')
    parser.add_argument('-p', '--program', required=True, help='The program to run and monitor, including any arguments, enclosed in quotes.')
    parser.add_argument('-m', '--memory-unit', choices=['MB', 'GB'], default='MB', help='The unit for memory usage (MB or GB). Default is MB.')
    parser.add_argument('-i', '--interval', type=float, default=1, help='The interval in seconds for measuring CPU and memory usage. Default is 1 second.')

    args = parser.parse_args()

    monitor_program(args.program, args.memory_unit, args.interval)

