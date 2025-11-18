
import argparse, json, os
import torch

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def to_tensor(mat):
    return torch.tensor(mat)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["keyerror","typeerror","indexerror","filenotfound","devicemismatch","ok"], default="keyerror")
    ap.add_argument("--json", default="data.json")
    args = ap.parse_args()

    if args.mode == "filenotfound":
        data = load_json("missing_data.json")  # FileNotFoundError
    else:
        data = load_json(args.json)

    if args.mode == "keyerror":
        values = data["valuesX"]   # KeyError
    else:
        values = data.get("values")

    if args.mode == "typeerror":
        values = {"oops": values}  # bad structure for torch.tensor
    elif args.mode == "indexerror":
        _ = values[10][0]          # IndexError

    x = to_tensor(values)

    if args.mode == "devicemismatch":
        cpu_tensor = x.to("cpu")
        if torch.cuda.is_available():
            gpu_tensor = x.to("cuda")
        else:
            gpu_tensor = x.to("cpu")
        y = cpu_tensor + gpu_tensor  # RuntimeError if devices differ

    print("Tensor shape:", x.shape)
    print("Done.")

if __name__ == "__main__":
    main()
