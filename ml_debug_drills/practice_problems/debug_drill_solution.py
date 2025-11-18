
import argparse, json, os
import torch

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def safe_get_values(data):
    if "values" in data:
        vals = data["values"]
    elif "vals" in data:
        vals = data["vals"]
    else:
        raise KeyError("Expected key 'values' (or fallback 'vals') not found in JSON.")
    return vals

def to_2d_float_tensor(values):
    if values is None:
        raise TypeError("values is None; cannot convert to tensor.")
    tensor = torch.tensor(values, dtype=torch.float32)
    if tensor.ndim != 2:
        tensor = tensor.reshape(-1, tensor.numel()) if tensor.numel() else tensor
    return tensor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="data.json")
    ap.add_argument("--device", default="cpu", choices=["cpu","cuda"])
    args = ap.parse_args()

    if not os.path.exists(args.json):
        raise FileNotFoundError(f"Input file not found: {args.json}")
    data = load_json(args.json)

    values = safe_get_values(data)
    x = to_2d_float_tensor(values)

    device = torch.device(args.device if (args.device=="cpu" or torch.cuda.is_available()) else "cpu")
    x = x.to(device)

    out = x.mean(dim=0)
    print("Device:", device)
    print("Tensor shape:", tuple(x.shape))
    print("Column means:", out.tolist())
    print("OK")

if __name__ == "__main__":
    main()
