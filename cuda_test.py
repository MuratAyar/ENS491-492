import torch

print("Torch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("cuDNN Enabled:", torch.backends.cudnn.enabled)
print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU Found")

x = torch.rand(3, 3).to("cuda")
print(x)