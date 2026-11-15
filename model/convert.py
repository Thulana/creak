# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["numpy", "h5py", "onnx"]
# ///
"""Build yamnet.onnx directly from the official yamnet.h5 weights (no TF, no tf2onnx).

Usage: uv run model/convert.py [cache_dir]
Writes yamnet.onnx and yamnet_class_map.csv next to this script.
Input  patches    f32 [1, 96, 64]   (one log-mel patch: frames x mel bands)
Output scores     f32 [1, 521]      (sigmoid)
       embeddings f32 [1, 1024]     (global average pool)
"""
import math
import shutil
import sys
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

H5_URL = "https://storage.googleapis.com/audioset/yamnet.h5"
CSV_URL = ("https://raw.githubusercontent.com/tensorflow/models/master/"
           "research/audioset/yamnet/yamnet_class_map.csv")
BN_EPS = 1e-4
FRAMES, BANDS = 96, 64
# (kind, stride, filters) per params in yamnet.py _YAMNET_LAYER_DEFS; all kernels are 3x3.
LAYERS = [("conv", 2, 32)] + [("sep", s, f) for s, f in [
    (1, 64), (2, 128), (1, 128), (2, 256), (1, 256), (2, 512),
    (1, 512), (1, 512), (1, 512), (1, 512), (1, 512), (2, 1024), (1, 1024)]]

here = Path(__file__).resolve().parent
cache = Path(sys.argv[1]) if len(sys.argv) > 1 else here / ".cache"
cache.mkdir(parents=True, exist_ok=True)


def fetch(url):
    dst = cache / url.rsplit("/", 1)[1]
    if not dst.exists():
        urllib.request.urlretrieve(url, dst)
    return dst


h5 = h5py.File(fetch(H5_URL), "r")
nodes, inits = [], []


def const(name, arr):
    inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=np.float32), name))
    return name


def same_pads(size, k, s):
    """TF 'same': extra pixel goes to the end. Returns (begin, end)."""
    total = max((math.ceil(size / s) - 1) * s + k - size, 0)
    return total // 2, total - total // 2


def conv_bn_relu(x, name, w, bn_prefix, stride, groups, hw):
    """Conv (BN folded into weights/bias) + Relu. w is in ONNX OIHW layout."""
    g = lambda n: h5[f"{bn_prefix}/{bn_prefix}/{n}:0"][()]
    scale = 1.0 / np.sqrt(g("moving_variance") + BN_EPS)
    w = w * scale[:, None, None, None]
    b = g("beta") - g("moving_mean") * scale
    k = w.shape[2]
    pt, pb = same_pads(hw[0], k, stride)
    pl, pr = same_pads(hw[1], k, stride)
    nodes.append(helper.make_node(
        "Conv", [x, const(f"{name}.w", w), const(f"{name}.b", b)], [f"{name}.conv"],
        name=f"{name}.conv", kernel_shape=[k, k], strides=[stride, stride],
        pads=[pt, pl, pb, pr], group=groups))
    nodes.append(helper.make_node("Relu", [f"{name}.conv"], [f"{name}.relu"], name=f"{name}.relu"))
    return f"{name}.relu", (math.ceil(hw[0] / stride), math.ceil(hw[1] / stride))


# [1,96,64] -> [1,1,96,64] (NCHW, one channel)
nodes.append(helper.make_node("Unsqueeze", ["patches", "unsq_axes"], ["x0"], name="unsqueeze"))
inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), "unsq_axes"))
x, hw, ch = "x0", (FRAMES, BANDS), 1
for i, (kind, stride, filters) in enumerate(LAYERS, 1):
    L = f"layer{i}"
    if kind == "conv":
        k = h5[f"{L}/conv/{L}/conv/kernel:0"][()]  # HWIO
        x, hw = conv_bn_relu(x, f"{L}.conv", k.transpose(3, 2, 0, 1), f"{L}/conv/bn", stride, 1, hw)
    else:
        k = h5[f"{L}/depthwise_conv/{L}/depthwise_conv/depthwise_kernel:0"][()]  # HWC1
        x, hw = conv_bn_relu(x, f"{L}.dw", k.transpose(2, 3, 0, 1), f"{L}/depthwise_conv/bn", stride, ch, hw)
        k = h5[f"{L}/pointwise_conv/{L}/pointwise_conv/kernel:0"][()]  # HWIO
        x, hw = conv_bn_relu(x, f"{L}.pw", k.transpose(3, 2, 0, 1), f"{L}/pointwise_conv/bn", 1, 1, hw)
    ch = filters

nodes.append(helper.make_node("GlobalAveragePool", [x], ["gap"], name="gap"))
nodes.append(helper.make_node("Flatten", ["gap"], ["embeddings"], name="flatten", axis=1))
nodes.append(helper.make_node(
    "Gemm", ["embeddings", const("logits.w", h5["logits/logits/kernel:0"][()]),
             const("logits.b", h5["logits/logits/bias:0"][()])], ["logits"], name="logits"))
nodes.append(helper.make_node("Sigmoid", ["logits"], ["scores"], name="sigmoid"))

graph = helper.make_graph(
    nodes, "yamnet",
    [helper.make_tensor_value_info("patches", TensorProto.FLOAT, [1, FRAMES, BANDS])],
    [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 521]),
     helper.make_tensor_value_info("embeddings", TensorProto.FLOAT, [1, 1024])],
    initializer=inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model, full_check=True)
onnx.save(model, here / "yamnet.onnx")
shutil.copyfile(fetch(CSV_URL), here / "yamnet_class_map.csv")
print("wrote", here / "yamnet.onnx", "final spatial", hw)
