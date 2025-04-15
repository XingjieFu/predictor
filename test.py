import torch
import os

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Count: {torch.cuda.device_count()}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

if torch.cuda.is_available():
    try:
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"Error accessing GPU: {e}")
else:
    print("CUDA is not available. Check driver, CUDA toolkit, or PyTorch installation.")