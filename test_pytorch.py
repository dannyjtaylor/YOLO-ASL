import torch

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA built version: {torch.version.cuda if hasattr(torch.version, "cuda") else "None"}')

if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU count: {torch.cuda.device_count()}')
else:
    print('No GPU detected')
    print('\nThis means PyTorch was installed without CUDA support.')
    print('You need to reinstall PyTorch with CUDA.')