import numpy as np
import enum


class DenseConstraint:

    def __init__(self, weights):
        assert len(weights.shape) == 2
        self.pos_w = np.maximum(weights, 0)
        self.neg_w = np.minimum(weights, 0)
        self.weights = weights

    def apply_weights(self, neg_weights, pos_weights):
        return self.pos_w @ pos_weights + self.neg_w @ neg_weights

    def apply_constraint(self, neg_cst, pos_cst):
        weights = self.pos_w @ pos_cst.weights + self.neg_w @ neg_cst.weights
        return DenseConstraint(weights)
        """
        if isinstance(neg_cst, OneToOneConstraint) and isinstance(neg_cst, OneToOneConstraint):
            weights = self.pos_w * pos_cst.weights + self.neg_w * neg_cst.weights
            return DenseConstraint(weights)
        elif isinstance(neg_cst, DenseConstraint) and isinstance(neg_cst, DenseConstraint):
            weights = self.pos_w @ pos_cst.weights + self.neg_w @ neg_cst.weights
            return DenseConstraint(weights)
        else:
            assert 0
        """

"""
class OneToOneConstraint:

    def __init__(self, weights):
        assert len(weights.shape) == 1
        self.pos_w = np.maximum(weights, 0)
        self.neg_w = np.minimum(weights, 0)

    def apply_weights(self, neg_weights, pos_weights):
        return self.pos_w * pos_weights + self.neg_w * neg_weights

    def apply_constraint(self, neg_cst, pos_cst):
        if isinstance(neg_cst, OneToOneConstraint) and isinstance(neg_cst, OneToOneConstraint):
            weights = self.pos_w * pos_cst.weights + self.neg_w * neg_cst.weights
            return OneToOneConstraint(weights)
        elif isinstance(neg_cst, DenseConstraint) and isinstance(neg_cst, DenseConstraint):
            weights = self.pos_w.reshape((-1, 1)) * pos_cst.weights + self.neg_w.reshape((-1, 1)) * neg_cst.weights
            return DenseConstraint(weights)
        else:
            assert 0
"""

class DeepPolyElement:
    def __init__(self, lower_cst, upper_cst, lower_bias, upper_bias):
        self.lower_cst = lower_cst
        self.upper_cst = upper_cst
        self.lower_bias = lower_bias
        self.upper_bias = upper_bias
