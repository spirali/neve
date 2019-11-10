import numpy as np

from neve import Input, ReLU, VerificationState, AffineTransformation
from neve.element import DenseConstraint
from neve.evaluate import bound_difference


def check_bounds(layer, inp, input_bounds, output_bounds, epsilon=0.01):
    np.random.seed(999)
    r = np.random.rand(250000, input_bounds[0].shape[0])
    values = ((r * (input_bounds[1] - input_bounds[0])) + input_bounds[0])
    output = layer.forward({inp: values})
    output_min = output.min(axis=0)
    output_max = output.max(axis=0)
    print("Omin", output_min)
    print("Omax", output_max)
    assert (output_bounds[0] <= output_min).all()
    assert (output_bounds[1] >= output_max).all()
    assert (output_bounds[0] + epsilon >= output_min).all()
    assert (output_bounds[1] - epsilon <= output_max).all()


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


def test_affine_transform():
    bounds = [
        np.array([2.0, -8.0, -3.0]),
        np.array([3.0, 2.0, -1.0])
    ]
    inp = Input()
    x = AffineTransformation(inp, np.array([
        [-1.0, 1.0, 3.0],
        [-1.0, 1.0, 0.0]
    ]),
        np.array([-4.0, 12.0],
    ))
    state = VerificationState({inp: bounds})
    r1, r2 = x.compute_bounds(state)
    assert (r1 == [-24.0, 1.0]).all()
    assert (r2 == [-7.0, 12.0]).all()


def test_input():
    bounds = [
        np.array([2.0, -8.0, -3.0]),
        np.array([3.0, 2.0, -1.0])
    ]
    inp = Input()
    state = VerificationState({inp: bounds})
    r = inp.compute_bounds(state)
    assert (r[0].shape == bounds[0].shape)
    assert (r[1].shape == bounds[1].shape)
    assert (r[0] == bounds[0]).all()
    assert (r[1] == bounds[1]).all()

    #cst = OneToOneConstraint(np.array([2, -4, 10]))
    #assert (inp.compute_upper_bounds(state, cst) == [6.0, 32.0, -10.0]).all()
    #assert (inp.compute_lower_bounds(state, cst) == [4.0, -8.0, -30.0]).all()

    cst = DenseConstraint(np.array([[1.0, 0, 0], [1.0, -3.0, 0.0], [-1.0, 1.0, 2.0]]))
    assert (inp.compute_lower_bounds(state, cst) == [2.0, -4.0, -17.0]).all()
    assert (inp.compute_upper_bounds(state, cst) == [3.0, 27.0, -2.0]).all()


def test_relu_element():
    bounds = [
        np.array([2.0, -8.0, -3.0]),
        np.array([3.0, 2.0, -1.0])
    ]
    inp = Input()
    x = ReLU(inp)
    state = VerificationState({inp: bounds})
    e = x.create_element(state)

    assert (e.upper_cst.weights == np.diag(np.array([1.0, 0.2, 0.0]))).all()
    assert (e.upper_bias == np.array([0.0, 1.6, 0.0])).all()

    assert (e.lower_cst.weights == np.diag(np.array([1.0, 0.0, 0.0]))).all()
    assert (e.lower_bias == np.array([0.0, 0.0, 0.0])).all()

    b1, b2 = x.compute_bounds(state)

    print(b1)

    assert (b1 == [2.0, 0.0, 0.0]).all()
    assert (b2 == [3.0, 2.0, 0.0]).all()

    # check_bounds(x, inp, bounds, output_bounds)


def test_simple_layer():
    bounds = [
        np.array([2.0, -8.0, -3.0]),
        np.array([3.0, 2.0, -1.0])
    ]
    inp = Input()
    x = AffineTransformation(inp, np.array([
        [-1.0, 1.0, 3.0],
        [-1.0, 1.0, 0.0],
        [2.0, 1.0, 3.0]
    ]),
        np.array([-4.0, 12.0, 0.0],
    ))
    x = ReLU(x)
    state = VerificationState({inp: bounds})
    b1, b2 = x.compute_bounds(state)
    assert (b1 == [0.0, 1.0, 0.0]).all()
    assert (b2 == [0.0, 12.0, 5.0]).all()


def test_three_layers():
    bounds = [
        np.array([2.0, -8.0, -3.0]),
        np.array([3.0, 2.0, -1.0])
    ]
    inp = Input()
    x = AffineTransformation(inp, np.array([
        [-1.0, 1.0, 3.0],
        [-1.0, 1.0, 0.0],
        [2.0, 1.0, 3.0],
        [0.0, 1.0, 0.0],
    ]),
        np.array([-4.0, 12.0, 0.0, 0.0],
    ))
    x = ReLU(x)

    x = AffineTransformation(x, np.array([
        [3.0, 1.0, -3.0, 1.0],
        [2.0, 2.0, 0.0, 1.0],
    ]),
        np.array([1.0, -1.0],
    ))
    x = ReLU(x)

    x = AffineTransformation(x, np.array([
        [5.0, 1.0],
        [2.0, -1.0],
        [1.0, -3.0]
    ]),
        np.array([-3.0, 100.0, -1.0],
    ))
    x = ReLU(x)

    state = VerificationState({inp: bounds})
    state.mode = "mode0"
    b1, b2 = x.compute_bounds(state)

    check_bounds(x, inp, bounds, (b1, b2), 10.0)

    state = VerificationState({inp: bounds})
    state.mode = "std"
    b1, b2 = x.compute_bounds(state)
    check_bounds(x, inp, bounds, (b1, b2), 35.0)


def test_random_layers():
    def random_at(shape):
        return AffineTransformation(x, np.random.random(shape) * 2 - 1.0, np.random.random(shape[0]) * 2 - 1.0)

    np.random.seed(320)
    r = np.random.random((10, 2)) - 0.5
    lo = np.min(r, axis=1)
    up = np.max(r, axis=1)
    bounds = [lo, up]

    inp = Input()
    x = inp
    x = random_at((20, 10))
    x = ReLU(x)

    x = random_at((15, 20))
    x = ReLU(x)

    x = random_at((6, 15))
    x = ReLU(x)

    x = random_at((5, 6))
    #x = ReLU(x)

    state = VerificationState({inp: bounds})
    b1, b2 = x.compute_bounds(state)
    print("-")
    print("LO", b1)
    print("UP", b2)

    check_bounds(x, inp, bounds, (b1, b2), 15.0)