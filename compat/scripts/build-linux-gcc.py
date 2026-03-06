#!/usr/bin/python3

"""
This script builds NovaPhy using the GCC compiler on Linux. 
It allows users to specify custom paths for GCC and CUDA 
installations, and optionally enables IPC support. The 
script sets up the necessary environment variables and CMake
 arguments before invoking pip to build the package.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Build NovaPhy with GCC compiler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--gcc-dir',
        type=str,
        default=None,
        help='Path to GCC installation directory',
        required=False
    )
    
    parser.add_argument(
        '--cuda-dir',
        type=str,
        default=None,
        help='Path to CUDA installation directory',
        required=False
    )

    parser.add_argument(
        '--ipc',
        action='store_true',
        help='enable IPC support'
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for debugging"
    )

    parser.add_argument(
        "--build-dir",
        type=str,
        default=r"build/{wheel_tag}",
        help="Directory to store build files"
    )

    parser.add_argument(
        "--cmake-standalone",
        action="store_true",
        help="Run cmake configuration step without scikit-build"
    )
    
    args = parser.parse_args()
    
    # Process arguments
    gcc_dir = args.gcc_dir
    cuda_dir = args.cuda_dir
    enable_ipc = args.ipc
    verbose_flag = args.verbose
    build_dir = args.build_dir
    cmake_standalone = args.cmake_standalone
    
    # Select preset based on IPC option
    if enable_ipc:
        cmake_args = ["--preset=ipc"]

    else:
        cmake_args = ["--preset=default"]

    if gcc_dir:
        gcc_path = os.path.join(gcc_dir, "bin", "gcc")
        gpp_path = os.path.join(gcc_dir, "bin", "g++")
        cmake_args.append(f"-DCMAKE_C_COMPILER={gcc_path}")
        cmake_args.append(f"-DCMAKE_CXX_COMPILER={gpp_path}")
        os.environ["CC"] = gcc_path
        os.environ["CXX"] = gpp_path
    else:
        # Use GCC in PATH
        os.environ["CC"] = "gcc"
        os.environ["CXX"] = "g++"

    if cuda_dir:
        nvcc_path = os.path.join(cuda_dir, "bin", "nvcc")
        cmake_args.append(f"-DCMAKE_CUDA_COMPILER={nvcc_path}")
        if gcc_dir:
            cmake_args.append(f"-DCMAKE_CUDA_HOST_COMPILER={gpp_path}")
            cuda_include = os.path.join(cuda_dir, "include")
            gcc_include = os.path.join(gcc_dir, "include")
            nvcc_flags = [f"--system-include={cuda_include},{gcc_include}"]
            cmake_args.append(f"-DCMAKE_CUDA_FLAGS=\"{' '.join(nvcc_flags)}\"")

    print("🧰Build configuration summary")
    print("CMAKE_ARGS: ", " ".join(cmake_args))
    print("CC:\t", os.environ["CC"])
    print("CXX:\t", os.environ["CXX"])

    if cmake_standalone:
        cmake_cmd = ["cmake", "-S", ".", "-B", build_dir] + cmake_args
        subprocess.run(cmake_cmd, check=True)
    else:
        os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = "8"
        os.environ["CMAKE_ARGS"] = " ".join(cmake_args)


        cmd = ["pip", "install", "-e", ".", "-C", f"build-dir={build_dir}"]
        if (verbose_flag):
            cmd.append("-v")
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
