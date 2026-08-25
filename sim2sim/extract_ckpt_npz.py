#!/usr/bin/env python3
"""Extract actor MLP + obs normalizer from a torch.save checkpoint (.pt)
WITHOUT torch — parse the zip container + data.pkl with stub classes.

Usage: extract_ckpt_npz.py <model.pt> <out.policy.npz>
"""
import io
import pickle
import re
import sys
import zipfile

import numpy as np

STORAGE_DTYPES = {
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "HalfStorage": np.float16,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
    "BFloat16Storage": np.uint16,  # raw 2-byte; only storage size matters
}


class StorageStub:
    def __init__(self, dtype: np.dtype, numel: int):
        self.dtype = dtype
        self.numel = numel
        self._bytes = None  # filled by loader


def rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None, *args):
    assert isinstance(storage, StorageStub)
    itemsize = storage.dtype.itemsize
    start = storage_offset * itemsize
    nbytes = int(np.prod(size)) * itemsize if len(size) else itemsize
    buf = storage._bytes[start:start + nbytes]
    # torch tensors are C-contiguous row-major; ndarray with custom strides
    return np.ndarray(
        shape=tuple(size),
        dtype=storage.dtype,
        buffer=buf,
        strides=tuple(s * itemsize for s in stride),
        order='C',
    )


def load_torch_zip(path: str) -> dict:
    zf = zipfile.ZipFile(path)
    names = zf.namelist()
    pkl_name = next(n for n in names if n.endswith("data.pkl"))
    storage_files = {}
    # entries look like archive/<key>/0 or <key>/0 depending on torch version
    for n in names:
        m = re.match(r"^(?:archive/)?(.+)/data/(\d+)$", n)
        if m:
            storage_files[m.group(2)] = n

    class U(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "_rebuild_tensor_v2" and module.endswith("_utils"):
                return rebuild_tensor_v2
            if name in STORAGE_DTYPES:
                return type(name, (), {"__module__": module})  # marker class
            if module == "collections" and name == "OrderedDict":
                from collections import OrderedDict
                return OrderedDict
            # generic fallback: return a dummy attribute container
            class Dummy:
                def __init__(self, *a, **k):
                    pass
                def __setstate__(self, state):
                    self.state = state
            return Dummy

        def persistent_load(self, pid):
            # pid: ('storage', storage_class, key, location, numel)
            typ = pid[0]
            assert typ == "storage", f"unexpected persistent id {pid!r}"
            storage_cls, key, _loc, numel = pid[1], pid[2], pid[3], pid[4]
            cls_name = storage_cls if isinstance(storage_cls, str) else getattr(storage_cls, "__name__", str(storage_cls))
            dtype = STORAGE_DTYPES.get(cls_name, np.uint8)
            st = StorageStub(np.dtype(dtype), numel)
            fname = storage_files.get(key)
            assert fname is not None, f"storage {key} not in archive ({list(storage_files)[:5]}...)"
            st._bytes = zf.read(fname)[: numel * dtype().itemsize if False else numel * np.dtype(dtype).itemsize]
            return st

    return U(io.BytesIO(zf.read(pkl_name))).load()


def main():
    ckpt_path, out_path = sys.argv[1], sys.argv[2]
    d = load_torch_zip(ckpt_path)
    sd = d.get("model_state_dict", d)
    if not isinstance(sd, dict):
        print("no model_state_dict; top keys:", list(d)[:20])
        sys.exit(2)
    print(f"[keys] {len(sd)} entries")
    actor_keys = [k for k in sd if "actor" in k.lower()]
    for k in actor_keys[:40]:
        v = sd[k]
        print("  ", k, getattr(v, "shape", type(v)))

    layers = {}
    for k, v in sd.items():
        m = re.match(r"actor\.(?:[a-zA-Z_]+\.)?(\d+)\.(weight|bias)$", k)
        if m:
            layers[(int(m.group(1)), m.group(2))] = np.asarray(v)
    out = {}
    idx = sorted({i for i, _ in layers})
    for n, i in enumerate(idx):
        out[f"l{n}_w"] = layers[(i, "weight")]
        out[f"l{n}_b"] = layers[(i, "bias")]
    for key_pref in ("actor_obs_normalizer", "obs_normalizer", "normalizer"):
        mk = f"{key_pref}._mean"
        sk = f"{key_pref}._std"
        if mk in sd and sk in sd:
            out["mean"] = np.asarray(sd[mk])
            out["std"] = np.asarray(sd[sk])
            break
    if not out or "l0_w" not in out or "mean" not in out:
        print("[FAIL] could not assemble policy layers/normalizer")
        sys.exit(3)
    np.savez(out_path, **out)
    shapes = [out[f"l{n}_w"].shape for n in range(len(idx))]
    print(f"[OK] saved {out_path}: layers {shapes}, mean {out['mean'].shape}, std {out['std'].shape}")


if __name__ == "__main__":
    main()
