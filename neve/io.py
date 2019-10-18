import json

from .layers import AffineTransformation, ReLU, Input

import numpy as np


def read_tf_file(filename):
    inp = Input()
    layer = inp
    with open(filename, "r") as f:
        while True:
            layer_type = f.readline()
            if not layer_type:
                break
            if layer_type != "ReLU\n":
                raise Exception("Not supported type", layer_type)

            weights = np.array(json.loads(f.readline()))
            bias = np.array(json.loads(f.readline()))
            layer = AffineTransformation(layer, weights, bias)
            layer = ReLU(layer)
    return layer, inp
