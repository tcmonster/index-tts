import torch

print("torch version:", torch.__version__)

# 检查 MPS 是否可用
if torch.backends.mps.is_available():
    print("✅ MPS is available")
else:
    print("❌ MPS is not available")

# 检查是否编译了 MPS 支持
if hasattr(torch.backends.mps, "is_built") and not torch.backends.mps.is_built():
    print("⚠️ MPS support is not built in this PyTorch install")

# 显示设备名称
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# 做一个简单操作
x = torch.ones((3,3), device=device)
y = x * 2
print("x:", x)
print("y:", y)