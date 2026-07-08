"""Export a MobileNetV4-Conv-Small float32 .tflite that matches the netkit
models/mobilenetv4_small.nk architecture (input 56x56x3, 10 classes).

Mirrors python/netkit/mobilenetv4_small.py block specs and src/mobilenetv4_uib.cpp
execution order so the TFLM benchmark runs the SAME op graph netkit runs. BN is
folded into conv (fixture random weights; accuracy is irrelevant, only op shapes
matter for latency). Padding stays inside the conv op ('same'/'valid') so no
extra PAD ops appear.
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

# (conv_bn, kernel, stride, out_ch) or (uib, start_dw, middle_dw, stride, out_ch, expand_ratio)
BLOCKS = [
    ("conv_bn", 3, 2, 32),
    ("conv_bn", 3, 2, 32),
    ("conv_bn", 1, 1, 32),
    ("conv_bn", 3, 2, 96),
    ("conv_bn", 1, 1, 64),
    ("uib", 5, 5, 2, 96, 3.0),
    ("uib", 0, 3, 1, 96, 2.0),
    ("uib", 0, 3, 1, 96, 2.0),
    ("uib", 0, 3, 1, 96, 2.0),
    ("uib", 0, 3, 1, 96, 2.0),
    ("uib", 3, 0, 1, 96, 4.0),
    ("uib", 3, 3, 2, 128, 6.0),
    ("uib", 5, 5, 1, 128, 4.0),
    ("uib", 0, 5, 1, 128, 4.0),
    ("uib", 0, 5, 1, 128, 3.0),
    ("uib", 0, 3, 1, 128, 4.0),
    ("uib", 0, 3, 1, 128, 4.0),
    ("conv_bn", 1, 1, 960),
]


def make_divisible(value, divisor=8):
    rounded = int(value + divisor / 2) // divisor * divisor
    result = max(divisor, rounded)
    if result < int(0.9 * value):
        result += divisor
    return result


def conv(x, filters, kernel, stride, relu):
    pad = "same" if kernel > 1 else "valid"
    x = layers.Conv2D(filters, kernel, strides=stride, padding=pad, use_bias=True,
                      activation="relu" if relu else None)(x)
    return x


def dwconv(x, kernel, stride, relu):
    pad = "same" if kernel > 1 else "valid"
    x = layers.DepthwiseConv2D(kernel, strides=stride, padding=pad, use_bias=True,
                               activation="relu" if relu else None)(x)
    return x


def uib(x, in_c, out_c, start_dw, middle_dw, stride, expand_ratio):
    expand_c = make_divisible(in_c * expand_ratio, 8)
    # middle_dw_downsample=1: stride lands on middle_dw when present, else start_dw.
    start_dw_stride = 1 if middle_dw > 0 else stride
    middle_dw_stride = stride
    residual = (stride == 1 and in_c == out_c)
    shortcut = x
    y = x
    if start_dw > 0:
        y = dwconv(y, start_dw, start_dw_stride, relu=False)   # + BN (folded)
    y = conv(y, expand_c, 1, 1, relu=True)                     # expand 1x1 + BN + ReLU
    if middle_dw > 0:
        y = dwconv(y, middle_dw, middle_dw_stride, relu=True)  # + BN + ReLU
    y = conv(y, out_c, 1, 1, relu=False)                       # project 1x1 + BN
    if residual:
        y = layers.Add()([shortcut, y])
    return y


def build(height=56, width=56, channels=3, num_classes=10):
    inp = layers.Input(shape=(height, width, channels))
    x = inp
    ch = channels
    for block in BLOCKS:
        if block[0] == "conv_bn":
            _, k, s, out_ch = block
            x = conv(x, out_ch, k, s, relu=True)
            ch = out_ch
        else:
            _, start_dw, middle_dw, s, out_ch, expand_ratio = block
            x = uib(x, ch, out_ch, start_dw, middle_dw, s, expand_ratio)
            ch = out_ch
    # head: global avg pool -> 1x1 conv to 1280 (relu) -> flatten -> dense(10)
    h, w = int(x.shape[1]), int(x.shape[2])
    x = layers.AveragePooling2D(pool_size=(h, w), strides=(h, w))(x)
    x = conv(x, 1280, 1, 1, relu=True)
    x = layers.Flatten()(x)
    out = layers.Dense(num_classes, activation=None)(x)
    return Model(inp, out, name="mobilenetv4_small")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    gen = os.path.join(here, "..", "generated")
    os.makedirs(gen, exist_ok=True)

    model = build()
    model.summary()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = []  # float32, no quantization
    tfl = converter.convert()

    out_path = os.path.join(gen, "mobilenetv4_small.tflite")
    with open(out_path, "wb") as f:
        f.write(tfl)
    print(f"wrote {out_path} ({len(tfl)} bytes)")

    # Report op set (for TFLM op resolver) and input/output shapes.
    interp = tf.lite.Interpreter(model_content=tfl)
    interp.allocate_tensors()
    print("input:", interp.get_input_details()[0]["shape"],
          interp.get_input_details()[0]["dtype"])
    print("output:", interp.get_output_details()[0]["shape"])
    ops = sorted({d["op_name"] for d in interp._get_ops_details()})
    print("ops:", ops)


if __name__ == "__main__":
    main()
