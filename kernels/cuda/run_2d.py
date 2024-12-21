import ctypes
import itertools
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import platform
import time
import os

def generate_tile_configs(max_shared_mem=49152):
    # Possible values for each parameter
    bm_values = [16, 32, 64, 128]
    bn_values = [16, 32, 64, 128]
    bk_values = [2, 4, 8, 16, 32]
    tm_values = [2, 4, 8, 16]
    tn_values = [2, 4, 8, 16]
    
    valid_configs = []
    
    for bm in bm_values:
        for bn in bn_values:
            for bk in bk_values:
                shared_mem_size = (bm * bk + bk * bn) * 4
                if shared_mem_size > max_shared_mem:
                    continue
                
                for tm in tm_values:
                    for tn in tn_values:
                        if bm % tm != 0 or bn % tn != 0:
                            continue
                            
                        threads_x = bn // tn
                        threads_y = bm // tm
                        total_threads = threads_x * threads_y
                        
                        if total_threads > 1024:
                            continue
                            
                        regs_per_thread = tm * tn
                        if regs_per_thread > 32:
                            continue
                            
                        valid_configs.append((bm, bn, bk, tm, tn))
    
    return valid_configs

def get_recommended_configs():
    return [
        (32, 32, 8, 4, 4),
        (32, 64, 8, 4, 8),
        (64, 32, 8, 8, 4),
        (64, 64, 8, 8, 8),
        (64, 64, 16, 8, 8),
    ]

def get_cuda_arch():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=gpu_name', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        gpu_name = result.stdout.strip()
        print(f"Detected GPU: {gpu_name}")
        
        if '4070' in gpu_name or '4080' in gpu_name or '4090' in gpu_name:
            return 'compute_89', 'sm_89'
        return 'compute_86', 'sm_86'
        
    except Exception as e:
        print(f"Warning: Could not detect GPU architecture: {e}")
        return 'compute_86', 'sm_86'

def compile_cuda_code():
    cuda_file = "matmul_2d_tiled.cu"
    virtual_arch, real_arch = get_cuda_arch()
    print(f"Compiling with virtual architecture {virtual_arch} and real architecture {real_arch}")
    
    lib_name = "libmatmul_2d_tiled.so"
    compile_cmd = [
        "nvcc", "-O3", "--shared",
        f"-arch={virtual_arch}",
        f"-code={real_arch}",
        "--ptxas-options=-v",
        "-Xcompiler", "-fPIC",
        "-o", lib_name,
        cuda_file
    ]
    
    print("Compiling CUDA code...")
    print(f"Command: {' '.join(compile_cmd)}")
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    
    if result.stdout:
        print("Compilation output:", result.stdout)
    if result.stderr:
        print("Compilation stderr:", result.stderr)
        
    if result.returncode != 0:
        raise RuntimeError("CUDA compilation failed")
    
    print("Compilation successful!")
    return os.path.abspath(lib_name)

def load_cuda_library(lib_path):
    try:
        print(f"Loading library from: {lib_path}")
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Library file not found: {lib_path}")
            
        if platform.system() != "Windows":
            os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + os.pathsep + os.path.dirname(lib_path)
        
        lib = ctypes.CDLL(lib_path)
        lib.run_matmul_benchmark.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,  # M, N, K
            ctypes.c_int, ctypes.c_int, ctypes.c_int,  # BM, BN, BK
            ctypes.c_int, ctypes.c_int,                # TM, TN
            ctypes.c_int,                              # verify
        ]
        lib.run_matmul_benchmark.restype = ctypes.c_float
        return lib
    except Exception as e:
        print(f"Error loading library: {e}")
        raise

def print_config_header():
    print("\n" + "="*140)
    print(f"{'Matrix (MxNxK)':<20} {'Workgroup':<15} {'Tile A':<15} {'Tile B':<15} {'Thread':<15} {'Performance':<20} {'Time'}")
    print(f"{'Dimensions':<20} {'(AxB)':<15} {'(BMxBK)':<15} {'(BKxBN)':<15} {'(TMxTN)':<15} {'(GFLOPS)':<20} {'(ms)'}")
    print("-"*140)

def print_config_result(M, N, K, workgroup_x, workgroup_y, BM, BN, BK, TM, TN, gflops, ms):
    matrix_dims = f"{M}x{N}x{K}"
    workgroup = f"{workgroup_x}x{workgroup_y}"
    tile_a = f"{BM}x{BK}"
    tile_b = f"{BK}x{BN}"
    thread_tile = f"{TM}x{TN}"
    print(f"{matrix_dims:<20} {workgroup:<15} {tile_a:<15} {tile_b:<15} {thread_tile:<15} {gflops:>8.2f} {ms:>8.2f}")

def run_benchmarks(lib, use_all_configs=False):
    matrix_sizes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]
    
    tile_sizes = generate_tile_configs() if use_all_configs else get_recommended_configs()
    results = []
    
    for M, N, K in matrix_sizes:
        print(f"\nMatrix Multiplication: A({M}x{K}) × B({K}x{N}) = C({M}x{N})")
        print_config_header()
        
        for BM, BN, BK, TM, TN in tile_sizes:
            workgroup_x = BN // TN
            workgroup_y = BM // TM
            
            try:
                ms = lib.run_matmul_benchmark(M, N, K, BM, BN, BK, TM, TN, 1)
                
                if ms < 0:
                    print(f"Config failed - BM={BM}, BN={BN}, BK={BK}, TM={TM}, TN={TN}")
                    continue
                
                flops = 2.0 * M * N * K
                gflops = (flops / (ms / 1000.0)) / 1e9
                
                print_config_result(M, N, K, workgroup_x, workgroup_y, BM, BN, BK, TM, TN, gflops, ms)
                
                results.append({
                    'M': M, 'N': N, 'K': K,
                    'BM': BM, 'BN': BN, 'BK': BK,
                    'TM': TM, 'TN': TN,
                    'Workgroup_X': workgroup_x,
                    'Workgroup_Y': workgroup_y,
                    'Time_ms': ms,
                    'GFLOPS': gflops
                })
                
            except Exception as e:
                print(f"Error running config: {e}")
    
    return pd.DataFrame(results)

def print_best_configs(results_df):
    print("\nBest Configurations by Matrix Size:")
    print("="*80)
    
    for size in results_df[['M', 'N', 'K']].drop_duplicates().values:
        M, N, K = size
        size_results = results_df[
            (results_df['M'] == M) & 
            (results_df['N'] == N) & 
            (results_df['K'] == K)
        ]
        best_config = size_results.loc[size_results['GFLOPS'].idxmax()]
        
        print(f"\nMatrix: A({M}x{K}) × B({K}x{N}) = C({M}x{N})")
        print(f"Best Configuration:")
        print(f"  {'Workgroup Size:':<20} {best_config['Workgroup_X']}x{best_config['Workgroup_Y']}")
        print(f"  {'Tile A Size:':<20} {best_config['BM']}x{best_config['BK']}")
        print(f"  {'Tile B Size:':<20} {best_config['BK']}x{best_config['BN']}")
        print(f"  {'Thread Tile:':<20} {best_config['TM']}x{best_config['TN']}")
        print(f"  {'Performance:':<20} {best_config['GFLOPS']:.2f} GFLOPS")
        print(f"  {'Time:':<20} {best_config['Time_ms']:.2f} ms")

def main():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(current_dir)
        
        lib_path = compile_cuda_code()
        if platform.system() != "Windows":
            lib_path = os.path.join(current_dir, os.path.basename(lib_path))
        
        lib = load_cuda_library(lib_path)
        
        print("\nStarting benchmarks...")
        results_df = run_benchmarks(lib, use_all_configs=False)
        
        results_df.to_csv('matmul_benchmark_results.csv', index=False)
        print("\nBenchmark results saved to 'matmul_benchmark_results.csv'")
        
        print_best_configs(results_df)
            
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()
