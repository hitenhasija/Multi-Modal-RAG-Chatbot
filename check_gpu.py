# check_gpu.py
import torch

if torch.cuda.is_available():
    print("CUDA is available! Here are the detected NVIDIA GPUs:")
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} NVIDIA GPU(s).")
    
    for i in range(num_gpus):
        print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        
    print("\nTo use the 3060, you will likely use the device index shown above.")
else:
    print("CUDA is not available. PyTorch cannot see your NVIDIA GPU.")
    print("Please ensure you have installed the GPU version of PyTorch and that your NVIDIA drivers are up to date.")