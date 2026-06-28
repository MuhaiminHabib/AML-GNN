import torch
import torch_geometric
import pandas as pd
import sklearn

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)

print("PyG:", torch_geometric.__version__)
print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)