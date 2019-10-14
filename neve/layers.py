import collections

import numpy as np

Constraint = collections.namedtuple("Constraint", ["weights", "bias"])


class DeepPolyElement:
    def __init__(self, lower_cst, upper_cst):
        assert upper_cst.weights.shape == lower_cst.weights.shape
        assert upper_cst.bias.shape == lower_cst.bias.shape
        assert upper_cst.weights.shape[0] == upper_cst.bias.shape[0]
        self.lower_cst = lower_cst
        self.upper_cst = upper_cst


class VerificationState:

    def __init__(self, inputs):
        self.bounds = inputs.copy()
        self.elements = {}

    def get_or_create_element(self, node):
        element = self.elements.get(node)
        if element is None:
            element = node.create_element(self)
            assert element is not None
            self.elements[node] = element
        return element


class Node:

    def __init__(self, parent):
        self.parent = parent

    def _make_constraints(self):
        raise NotImplementedError

    def create_element(self, state):
        raise NotImplementedError()

    def compute_bounds(self, state):
        element = state.get_or_create_element(self)
        print(element.lower_cst.weights)
        bounds = (self.parent.compute_lower_bounds(state, element.lower_cst.weights) + element.lower_cst.bias,
                  self.parent.compute_upper_bounds(state, element.upper_cst.weights) + element.upper_cst.bias)
        state.bounds[self] = bounds
        return bounds

    def compute_upper_bounds(self, state, weights):
        element = state.get_or_create_element(self)

        pos_w = np.maximum(weights, 0)
        neg_w = np.minimum(weights, 0)
        new_weights = pos_w @ element.upper_cst.weights + neg_w @ element.lower_cst.weights
        bounds = pos_w @ element.upper_cst.bias + neg_w @ element.lower_cst.bias
        return bounds + self.parent.compute_upper_bounds(new_weights, state)

    def compute_lower_bounds(self, state, weights):
        element = state.get_or_create_element(self)

        pos_w = np.maximum(weights, 0)
        neg_w = np.minimum(weights, 0)

        new_weights = pos_w @ element.lower_cst.weights + neg_w @ element.upper_cst.weights
        bounds = pos_w @ element.lower_cst.bias + neg_w @ element.upper_cst.bias
        return bounds + self.parent.compute_weighted_bounds(new_weights, state)


class Input:

    def compute_bounds(self, state):
        bounds = state.bounds[self]
        return bounds

    def compute_upper_bounds(self, state, weights):
        bounds = state.bounds[self]
        pos_w = np.maximum(weights, 0)
        neg_w = np.minimum(weights, 0)
        return pos_w @ bounds[1] + neg_w @ bounds[0]

    def compute_lower_bounds(self, state, weights):
        bounds = state.bounds[self]
        pos_w = np.maximum(weights, 0)
        neg_w = np.minimum(weights, 0)
        return pos_w @ bounds[0] + neg_w @ bounds[1]

    def forward(self, inputs):
        return inputs[self]


class AffineTransformation(Node):

    def __init__(self, parent, weights, bias):
        super().__init__(parent)
        self.weights = weights
        self.bias = bias

    def create_element(self, state):
        cst = Constraint(self.weights, self.bias)
        return DeepPolyElement(cst, cst)

    def forward(self, inputs):
        return self.parent.forward(inputs) @ self.weights.transpose() + self.bias


class ReLU(Node):

    def forward(self, inputs):
        return np.maximum(self.parent.forward(inputs), 0.0)

    def create_element(self, state):
        lo_bounds, up_bounds = self.parent.compute_bounds(state)

        n = lo_bounds.shape[0]
        up_w = np.zeros((n, 1))
        lo_w = np.zeros((n, 1))
        up_bias = np.zeros((n,))
        lo_bias = np.zeros((n,))

        linear = lo_bounds >= 0
        up_w[linear] = 1.0
        lo_w[linear] = 1.0

        mixed = (up_bounds > 0) & ~linear

        m_lo = lo_bounds[mixed]
        m_up = up_bounds[mixed]

        lambda_ = m_up / (m_up - m_lo)
        up_w[mixed] = lambda_
        up_bias[mixed] = -m_lo * lambda_

        up_cst = Constraint(up_w, up_bias)
        lo_cst = Constraint(lo_w, lo_bias)

        return DeepPolyElement(lo_cst, up_cst)
