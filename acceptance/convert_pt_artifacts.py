#!/usr/bin/env python3
"""Convert v25 acceptance .pt artifacts into viewable formats.

Works WITHOUT torch: these .pt files are raw pickle dumps. If a file happens
to be a real torch zip container and torch is installed, torch.load is used.

Usage:
    python3 acceptance/convert_pt_artifacts.py            # unpack all
    python3 acceptance/convert_pt_artifacts.py --npz      # also emit per-motion .npz
    python3 acceptance/convert_pt_artifacts.py FILE OUT   # convert a single file
"""
import argparse
import io
import json
import os
import pickle
import sys
import zipfile

DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v25_artifacts")
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v25_unpacked")

TORCH_ZIP_MAGIC = b"PK\x03\x04"


def smart_load(path):
    """Load a .pt file with pickle, or torch.load if it is a zip container."""
    with open(path, "rb") as f:
        head = f.read(4)
    if head == TORCH_ZIP_MAGIC:
        try:
            import torch
            return torch.load(path, map_location="cpu", weights_only=False)
        except ImportError:
            raise RuntimeError(
                f"{path} is a torch zip container but torch is not installed; "
                "pip install torch to read it"
            )
    with open(path, "rb") as f:
        return pickle.load(f)


def to_numpy(x):
    """Tensor/ndarray -> ndarray; recursive for list/dict."""
    import numpy as np
    if hasattr(x, "detach"):  # torch tensor
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)):
        return [to_numpy(i) for i in x]
    if isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    return x


def convert_report(obj, out_path):
    """model_retarget_report.pt: dict with 'report' = JSON string."""
    if isinstance(obj, dict) and isinstance(obj.get("report"), str):
        with open(out_path, "w") as f:
            json.dump(json.loads(obj["report"]), f, indent=1, ensure_ascii=False)
    else:  # fallback: dump whole dict
        with open(out_path, "w") as f:
            json.dump(to_numpy(obj), f, indent=1, ensure_ascii=False, default=str)
    print(f"  -> {out_path}")


def convert_data(obj, out_dir, emit_npz=False):
    """model_retarget_data.pt: dict {relative/path.pkl: pickled-bytes, *.json: bytes}."""
    import numpy as np

    n_pkl = n_json = n_npz = 0
    for rel, payload in obj.items():
        rel = rel.replace("\\", "/").lstrip("/")
        dest = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        if rel.endswith(".json"):
            if isinstance(payload, bytes):
                # sanity: pretty-print
                json.dump(json.loads(payload), open(dest, "w"),
                          indent=1, ensure_ascii=False)
            else:
                json.dump(to_numpy(payload), open(dest, "w"),
                          indent=1, ensure_ascii=False, default=str)
            n_json += 1
            print(f"  -> {dest}")
        elif rel.endswith(".pkl"):
            if isinstance(payload, bytes):
                with open(dest, "wb") as f:
                    f.write(payload)
            else:  # already a dict (torch.load case)
                with open(dest, "wb") as f:
                    pickle.dump(to_numpy(payload), f)
            n_pkl += 1
            if emit_npz:
                motion = payload if isinstance(payload, dict) else pickle.load(
                    open(dest, "rb"))
                npz_path = dest[:-4] + ".npz"
                arrays = {k: to_numpy(v) for k, v in motion.items()
                          if hasattr(v, "shape")}
                np.savez_compressed(npz_path, **arrays)
                n_npz += 1
        else:
            with open(dest, "wb") as f:
                f.write(payload if isinstance(payload, bytes)
                        else pickle.dumps(to_numpy(payload)))
            print(f"  -> {dest}")
    print(f"  unpacked: {n_pkl} .pkl, {n_json} .json"
          + (f", {n_npz} .npz" if emit_npz else ""))
    return n_pkl, n_json


def summarize(out_dir):
    """Print one line per motion so the result is human-checkable."""
    import numpy as np
    print("\n  motion summary (name | fps | frames | dof | root_z range):")
    for root, _, files in sorted(os.walk(out_dir)):
        for fn in sorted(files):
            if not fn.endswith(".pkl"):
                continue
            m = pickle.load(open(os.path.join(root, fn), "rb"))
            dof = getattr(m.get("dof_pos"), "shape", None)
            rz = np.asarray(m["root_pos"])[:, 2]
            print(f"    {os.path.relpath(os.path.join(root, fn), out_dir):40s} "
                  f"| {m['fps']:5.0f} | {m['root_pos'].shape[0]:5d} | "
                  f"{dof[1] if dof else '?':2d} | [{rz.min():.3f},{rz.max():.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", help="input .pt files (default: all in v25_artifacts/)")
    ap.add_argument("-o", "--out", default=DEFAULT_OUT, help="output dir")
    ap.add_argument("--npz", action="store_true", help="also emit per-motion .npz")
    args = ap.parse_args()

    inputs = args.inputs
    if not inputs:
        inputs = sorted(
            os.path.join(DEFAULT_DIR, f)
            for f in os.listdir(DEFAULT_DIR) if f.endswith(".pt")
        ) if os.path.isdir(DEFAULT_DIR) else []
    if not inputs:
        sys.exit("no input .pt files found")

    os.makedirs(args.out, exist_ok=True)
    for path in inputs:
        name = os.path.basename(path)
        print(f"\n== {name}")
        obj = smart_load(path)

        if "report" in name or ("data" not in name and isinstance(obj, dict)
                                and isinstance(obj.get("report"), str)):
            convert_report(obj, os.path.join(args.out, name.replace(".pt", ".json")))
        elif "data" in name or any(k.endswith(".pkl") for k in (obj or {})):
            convert_data(obj, args.out, emit_npz=args.npz)
        else:  # e.g. pipeline_meta.pt — just dump as json
            out = os.path.join(args.out, name.replace(".pt", ".json"))
            json.dump(to_numpy(obj), open(out, "w"), indent=1, ensure_ascii=False, default=str)
            print(f"  -> {out}")

    if any("data" in os.path.basename(p) for p in inputs):
        summarize(args.out)
    print("\nDONE ->", args.out)


if __name__ == "__main__":
    main()
