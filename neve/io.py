import json

import numpy as np


def read_tf_file(filename):
    with open(filename, "r") as f:
        while True:
            layer_type = f.readline()
            if not layer_type:
                break
            if layer_type != "ReLU\n":
                raise Exception("Not supported type", layer_type)

            weights = np.array(json.loads(f.readline()))
            print(weights)

            bias = np.array(json.loads(f.readline()))
            print(bias)
