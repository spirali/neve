import tensorflow as tf

from .element import DeepPolyElement
from .matrix import FullMatrix, DiagonalMatrix


class VerificationState:

    def __init__(self, inputs):
        self.bounds = inputs.copy()
        self.elements = {}
        self.mode = "std"

    def get_or_create_element(self, node):
        element = self.elements.get(node)
        if element is None:
            element = node.create_element(self)
            assert element is not None
            self.elements[node] = element
        return element


def _check_bounds_invariant(bounds):
    assert len(bounds) == 2
    assert len(bounds[0].shape) == 1
    assert bounds[0].shape == bounds[1].shape


class Node:

    def __init__(self, parent):
        self.parent = parent

    def create_element(self, state):
        raise NotImplementedError()

    def compute_bounds(self, state):
        element = state.get_or_create_element(self)
        bounds = (self.parent.compute_lower_bounds(state, element.lower_cst) + element.lower_bias,
                  self.parent.compute_upper_bounds(state, element.upper_cst) + element.upper_bias)
        _check_bounds_invariant(bounds)
        state.bounds[self] = bounds
        return bounds

    def compute_upper_bounds(self, state, constraint):
        element = state.get_or_create_element(self)
        neg = constraint.negative()
        pos = constraint.positive()
        bounds = neg.tensordot(element.lower_bias) + pos.tensordot(element.upper_bias)
        new_constraint = neg @ element.lower_cst + pos @ element.upper_cst

        assert len(bounds.shape) == 1
        result = bounds + self.parent.compute_upper_bounds(state, new_constraint)
        assert len(result.shape) == 1
        return result

    def compute_lower_bounds(self, state, constraint):
        element = state.get_or_create_element(self)
        neg = constraint.negative()
        pos = constraint.positive()
        bounds = pos.tensordot(element.lower_bias) + neg.tensordot(element.upper_bias)
        new_constraint = pos @ element.lower_cst + neg @ element.upper_cst
        return bounds + self.parent.compute_lower_bounds(state, new_constraint)


class Input:

    def compute_bounds(self, state):
        bounds = state.bounds[self]
        _check_bounds_invariant(bounds)
        return bounds

    def compute_upper_bounds(self, state, constraint):
        bounds = state.bounds[self]
        _check_bounds_invariant(bounds)
        neg = constraint.negative()
        pos = constraint.positive()
        result = neg.tensordot(bounds[0]) + pos.tensordot(bounds[1])
        assert len(result.shape) == 1
        return result

    def compute_lower_bounds(self, state, constraint):
        bounds = state.bounds[self]
        _check_bounds_invariant(bounds)
        neg = constraint.negative()
        pos = constraint.positive()
        result = neg.tensordot(bounds[1]) + pos.tensordot(bounds[0])
        assert len(result.shape) == 1
        return result

    def forward(self, inputs):
        return inputs[self]


class AffineTransformation(Node):

    def __init__(self, parent, weights, bias):
        super().__init__(parent)
        self.weights = weights
        self.bias = bias

    def forward(self, inputs):
        return self.parent.forward(inputs) @ tf.transpose(self.weights) + self.bias

    def create_element(self, state):
        cst = FullMatrix(self.weights)
        bias = self.bias
        return DeepPolyElement(cst, cst, bias, bias)

    """
    def forward(self, inputs):
        return self.parent.forward(inputs) @ self.weights.transpose() + self.bias
    """


class ReLU(Node):

    def forward(self, inputs):
        return tf.maximum(self.parent.forward(inputs), 0.0)

    def create_element(self, state):
        lo_bounds, up_bounds = self.parent.compute_bounds(state)

        pos = tf.maximum(up_bounds, 0)
        neg = tf.minimum(lo_bounds, 0)
        lambda_ = pos / (pos - neg)

        n = lo_bounds.shape[0]
        lo_bias = tf.zeros((n,))
        up_bias = -neg * lambda_

        up_w = lambda_

        print(lambda_.dtype)

        if state.mode == "std":
            lo_w = tf.round(lambda_)
        elif state.mode == "mode0":
            lo_w = tf.dtypes.cast(lambda_ >= 0.99999, tf.float32)
        elif state.mode == "mode1":
            lo_w = tf.dtypes.cast(lambda_ >= 0.00001, tf.float32)
        else:
            raise Exception("Unknown mode '{}'".format(state.mode))

        up_cst = DiagonalMatrix(up_w)
        lo_cst = DiagonalMatrix(lo_w)
        return DeepPolyElement(lo_cst, up_cst, lo_bias, up_bias)
