import collections

import numpy as np
from .element import DenseConstraint, DeepPolyElement


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


class Node:

    def __init__(self, parent):
        self.parent = parent

    def create_element(self, state):
        raise NotImplementedError()

    def compute_bounds(self, state):
        element = state.get_or_create_element(self)
        bounds = (self.parent.compute_lower_bounds(state, element.lower_cst) + element.lower_bias,
                  self.parent.compute_upper_bounds(state, element.upper_cst) + element.upper_bias)
        state.bounds[self] = bounds
        return bounds

    def compute_upper_bounds(self, state, constraint):
        element = state.get_or_create_element(self)
        bounds = constraint.apply_weights(element.lower_bias, element.upper_bias)
        new_constraint = constraint.apply_constraint(element.lower_cst, element.upper_cst)
        return bounds + self.parent.compute_upper_bounds(state, new_constraint)

    def compute_lower_bounds(self, state, constraint):
        element = state.get_or_create_element(self)
        bounds = constraint.apply_weights(element.upper_bias, element.lower_bias)
        new_constraint = constraint.apply_constraint(element.upper_cst, element.lower_cst)
        return bounds + self.parent.compute_lower_bounds(state, new_constraint)


class Input:

    def compute_bounds(self, state):
        bounds = state.bounds[self]
        return bounds

    def compute_upper_bounds(self, state, constraint):
        bounds = state.bounds[self]
        return constraint.apply_weights(bounds[0], bounds[1])

    def compute_lower_bounds(self, state, constraint):
        bounds = state.bounds[self]
        return constraint.apply_weights(bounds[1], bounds[0])

    def forward(self, inputs):
        return inputs[self]


class AffineTransformation(Node):

    def __init__(self, parent, weights, bias):
        super().__init__(parent)
        self.weights = weights
        self.bias = bias

    def forward(self, inputs):
        return self.parent.forward(inputs) @ self.weights.transpose() + self.bias

    def create_element(self, state):
        cst = DenseConstraint(self.weights)
        bias = self.bias
        return DeepPolyElement(cst, cst, bias, bias)

    """
    def forward(self, inputs):
        return self.parent.forward(inputs) @ self.weights.transpose() + self.bias
    """


class ReLU(Node):

    def forward(self, inputs):
        return np.maximum(self.parent.forward(inputs), 0.0)

    def create_element(self, state):
        lo_bounds, up_bounds = self.parent.compute_bounds(state)

        pos = np.maximum(up_bounds, 0)
        neg = np.minimum(lo_bounds, 0)
        lambda_ = pos / (pos - neg)

        n = lo_bounds.shape[0]
        lo_bias = np.zeros((n,))
        up_bias = -neg * lambda_

        up_w = lambda_

        if state.mode == "std":
            lo_w = np.round(lambda_)
        elif state.mode == "mode0":
            lo_w = (lambda_ >= 0.99999)
        elif state.mode == "mode1":
            lo_w = (lambda_ >= 0.00001)
        else:
            raise Exception("Unknown mode '{}'".format(state.mode))

        up_cst = DenseConstraint(np.diag(up_w))
        lo_cst = DenseConstraint(np.diag(lo_w))
        return DeepPolyElement(lo_cst, up_cst, lo_bias, up_bias)
    