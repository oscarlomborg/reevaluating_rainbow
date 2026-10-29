
import torch

path = "model.pt"
checkpoint = torch.load(path, map_location="cpu", weights_only=False)

def inspect(obj, prefix="", depth=0, max_depth=5):
    if isinstance(obj, torch.Tensor):
        print(f"{prefix}: Tensor {tuple(obj.shape)}")

    elif isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if depth < max_depth:
                inspect(value, name, depth + 1, max_depth)
            else:
                print(f"{name}: {type(value).__name__}")

    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            if depth < max_depth:
                inspect(value, f"{prefix}[{i}]", depth + 1, max_depth)

    else:
        print(f"{prefix}: {type(obj).__name__} = {str(obj)[:100]}")

print("Checkpoint type:", type(checkpoint))
print("\n=== CHECKPOINT CONTENTS ===")
inspect(checkpoint)
