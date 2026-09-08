#!/usr/bin/env python3
"""Export the rsl_rl actor MLP (from .policy.npz) to ONNX for real-robot deployment.

Graph: input(288) -> [Linear -> ELU] x3 -> Linear -> output(29), with the
observation normalizer fused as (x - mean) / std *before* the first layer
(v26 sim2sim verified this exact forward path in numpy; ONNX must match it
bit-for-bit, which we check against mujoco_rollout.run_policy).

Usage:
  python sim2sim/export_onnx.py model_3999.policy.npz x1_policy.onnx \
      [--meta meta.json]
"""
import argparse
import functools
import json
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

IN_DIM, OUT_DIM = 288, 29


def build_onnx(layers, mean, std, out_path, opset=17):
    import onnx
    from onnx import helper, TensorProto as T

    inits, nodes = [], []
    X = "obs"

    def init(name, arr):
        t = helper.make_tensor(name, T.FLOAT, arr.shape, arr.astype(np.float32).flatten())
        inits.append(t)
        return name

    # fused normalizer: xn = (obs - mean) / std
    init("mean", mean.reshape(1, IN_DIM))
    init("std", std.reshape(1, IN_DIM))
    nodes.append(helper.make_node("Sub", [X, "mean"], ["xn0"]))
    nodes.append(helper.make_node("Div", ["xn0", "std"], ["xn"]))

    cur = "xn"
    for i, (w, b) in enumerate(layers):
        wn = init(f"w{i}", w.T.astype(np.float32))       # Linear: y = x @ W^T + b
        out = f"h{i}" if i < len(layers) - 1 else "action"
        ins = [cur, wn] + ([init(f"b{i}", b.astype(np.float32))] if b is not None else [])
        nodes.append(helper.make_node("MatMul" if b is None else "Gemm",
                                      ins if b is None else [cur, wn, f"b{i}"],
                                      [f"mm{i}"]))
        if i < len(layers) - 1:                           # ELU(x) = max(0,x)+min(0,exp(x)-1)
            nodes.append(helper.make_node("Elu", [f"mm{i}"], [out], alpha=1.0))
        else:
            nodes.append(helper.make_node("Identity", [f"mm{i}"], [out]))
        cur = out

    graph = helper.make_graph(
        nodes, "x1_actor",
        [helper.make_tensor_value_info(X, T.FLOAT, [1, IN_DIM])],
        [helper.make_tensor_value_info("action", T.FLOAT, [1, OUT_DIM])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)],
                              producer_name="X1_29_AMP")
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, out_path)


def main():
    import onnxruntime as ort

    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("out")
    ap.add_argument("--meta", default=None)
    args = ap.parse_args()

    z = np.load(args.npz)
    layers, i = [], 0
    while f"l{i}_w" in z.files:
        layers.append((z[f"l{i}_w"].astype(np.float32),
                       z[f"l{i}_b"].astype(np.float32) if f"l{i}_b" in z.files else None))
        i += 1
    mean = z["mean"].astype(np.float32).reshape(1, IN_DIM)
    std = z["std"].astype(np.float32).reshape(1, IN_DIM)
    build_onnx(layers, mean, std, args.out)

    # verify against the numpy reference (same math as mujoco_rollout.run_policy)
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        obs = rng.normal(0, 1, (1, IN_DIM)).astype(np.float32) * 3
        ref = obs
        x = (obs - mean) / std
        for w, b in layers[:-1]:
            x = x @ w.T + (b if b is not None else 0)
            x = np.where(x < 0, np.expm1(np.minimum(x, 20)), x)
        w, b = layers[-1]
        ref = x @ w.T + (b if b is not None else 0)
        got = sess.run(["action"], {"obs": obs})[0]
        worst = max(worst, float(np.abs(ref - got).max()))
    print(f"[OK] {args.out}: 200 random-obs max |onnx - numpy| = {worst:.2e}")
    meta = {"src_npz": str(args.npz), "in_dim": IN_DIM, "out_dim": OUT_DIM,
            "hidden": [int(w.shape[0]) for w, _ in layers],
            "activation": "elu", "normalizer": "fused (x-mean)/std",
            "verify_max_abs_err": worst}
    if args.meta:
        Path(args.meta).write_text(json.dumps(meta, indent=1))
    print(f"[META] {json.dumps(meta)}")


if __name__ == "__main__":
    main()
