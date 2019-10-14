import numpy as np

from neve import Input, ReLU, VerificationState


def check_bounds(layer, inp, input_bounds, output_bounds, epsilon=0.01):
    g = np.random.seed(1002)
    r = g.rand(100000, input_bounds[0].shape[0])
    values = ((r * (input_bounds[1] - input_bounds[0])) + input_bounds[0])
    output = layer.forward({inp: values})
    output_min = output.min(axis=0)
    output_max = output.max(axis=0)
    assert (output_bounds[0] <= output_min).all()
    assert (output_bounds[1] >= output_max).all()
    assert (output_bounds[0] + epsilon >= output_min).all()
    assert (output_bounds[1] + epsilon <= output_max).all()


"""
def test_affine():
    inp = Input()
    x = inp
    x = AffineTransformation(x,
        np.array([[1, -2, 3],
                  [-1, -1, -2],
                  [1, 1, 1],
                  [0, 0, 1]]), np.array([2, -1, -4, 0]))
    x = ReLU(x)

    bounds = [
        np.array([-1, -2, 3]),
        np.array([2, 3, 10]),
    ]

    r = np.random.rand(100000, 3)
    rr = ((r * (bounds[1] - bounds[0])) + bounds[0])
    dd = x.forward({inp: rr})
    #dd = r
    #print("MM", dd.min(axis=0), dd.max(axis=0))

    #state = VerificationState({inp: bounds})
    #print(x.compute_bounds(state))

    #print(inp.compute_bounds(state))
"""


def test_relu_element():
    bounds = [
        np.array([2.0, -8.0, -3.0]),
        np.array([3.0, 2.0, -1.0])
    ]
    inp = Input()
    x = ReLU(inp)
    state = VerificationState({inp: bounds})
    e = x.create_element(state)

    assert (e.upper_cst.weights == np.array([[1.0], [0.2], [0.0]])).all()
    assert (e.upper_cst.bias == np.array([0.0, 1.6, 0.0])).all()

    assert (e.lower_cst.weights == np.array([[1.0], [0.0], [0.0]])).all()
    assert (e.lower_cst.bias == np.array([0.0, 0.0, 0.0])).all()

    # output_bounds = x.compute_bounds(state)
    # check_bounds(x, inp, bounds, output_bounds)
