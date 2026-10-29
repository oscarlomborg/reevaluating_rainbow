
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

def count_parameters(obj):
    if isinstance(obj, torch.nn.Module):
        return sum(p.numel() for p in obj.parameters())

    elif isinstance(obj, torch.Tensor):
        return obj.numel()

    elif isinstance(obj, dict):
        return sum(count_parameters(v) for v in obj.values())

    elif isinstance(obj, (list, tuple)):
        return sum(count_parameters(v) for v in obj)

    return 0

print("Checkpoint type:", type(checkpoint))
print("\n=== CHECKPOINT CONTENTS ===")
inspect(checkpoint)

# Select the model weights rather than optimizer state
model = checkpoint

if isinstance(checkpoint, dict):
    for key in ("model_state_dict", "state_dict", "model"):
        if key in checkpoint:
            model = checkpoint[key]
            break

total_params = count_parameters(model)

print("\n=== MODEL PARAMETERS ===")
print(f"Total parameters: {total_params:,}")
print(f"Parameters (millions): {total_params / 1e6:.4f} M")
print(f"Estimated FP32 weight memory: {total_params * 4 / 1024**2:.3f} MiB")