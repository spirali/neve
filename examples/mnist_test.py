import numpy as np

import neve


def main():
    net, inp = neve.io.read_tf_file("mnist_relu_3_50.tf")

    data = np.load("mnist.npz")
    x_test = data["x_test"].astype(np.float32)
    y_test = data["y_test"]

    result = net.forward({inp: x_test.reshape(-1, 784)})

    b = x_test[1].reshape(784)
    e = 10
    lo = np.maximum(b - e, 0)
    up = np.minimum(b + e, 255)

    result = net.forward({inp: [b, lo, up]})
    print("B = ", result[0])
    print("LO = ", result[1])
    print("UP = ", result[2])

    print(lo)

    state = neve.VerificationState({inp: [lo, up]})
    r_lo, r_up = net.compute_bounds(state)
    print(r_lo)
    print(r_up)

    # print(result[1])
    # print(y_test[1])


if __name__ == "__main__":
    main()
