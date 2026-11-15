"""CUDA and PyTorch Hardware Verification Script.

Inspects PyTorch CUDA availability, GPU device properties, and executes
a small GPU tensor operation if PyTorch is installed.
"""

import sys


def main():
    print("================================================================================")
    print("PYTORCH CUDA HARDWARE VERIFICATION")
    print("================================================================================\n")

    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"CUDA Available in PyTorch: {cuda_avail}")

        if cuda_avail:
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            print(f"Device Count: {gpu_count}")
            print(f"Current Device Index: {current_device}")
            print(f"GPU Device Name: {gpu_name}")

            # Execute GPU tensor test
            print("\nExecuting test tensor operation on CUDA...")
            x = torch.randn(1024, 1024, device="cuda")
            y = torch.matmul(x, x)
            print(f"CUDA Matrix Multiplication Successful: Output shape = {y.shape}, Device = {y.device}")
            print("Status: PYTORCH CUDA ACCELERATION OPERATIONAL")
        else:
            print("Status: CUDA is UNAVAILABLE in the installed PyTorch package.")
            print("Note: If neural auditor training is required, install torch with CUDA support:")
            print("pip install torch --index-url https://download.pytorch.org/whl/cu124")

    except ImportError:
        print("Status: PyTorch is not installed in the active environment.")
        print("Note: Local LLM inference uses Ollama native NVIDIA CUDA backend directly.")
        print("If PyTorch neural components are needed, install torch with CUDA support:")
        print("pip install torch --index-url https://download.pytorch.org/whl/cu124")

    print("\n================================================================================")


if __name__ == "__main__":
    main()
