# gpu_diagnostic.py
import torch
import os
from dotenv import load_dotenv

load_dotenv()

print("=== GPU DIAGNOSTIC ===")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

print(f"\nEnvironment Variables:")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")

# Test GPU computation
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    z = torch.mm(x, y)
    print(f"GPU computation test: SUCCESS")
    print(f"Result device: {z.device}")
else:
    print("GPU computation test: FAILED - No GPU detected")