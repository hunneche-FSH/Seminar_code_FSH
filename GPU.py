import torch

x = torch.rand(5, 3)
print(x)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("❌ Still on CPU. Check installation.")

    torch.cuda.is_available()