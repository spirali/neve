#import numpy as np
import tensorflow as tf
import enum


def _apply_dense(weights, cst):
    if isinstance(cst, DenseConstraint):
        return weights @ cst.weights
    elif isinstance(cst, OneToOneConstraint):
        return tf.tensordot(weights, cst.weights, 1)
    else:
        raise Exception("Invalid constraint")

def _apply_one_to_one(weights, cst):
    if isinstance(cst, DenseConstraint):
        return weights

class DenseConstraint:

    def __init__(self, weights):
        #print("WEIGHTS SHAPE", weights.shape)
        assert len(weights.shape) == 2
        self.pos_w = tf.maximum(weights, 0)
        self.neg_w = tf.minimum(weights, 0)
        self.weights = weights

    def apply_weights(self, neg_weights, pos_weights):
        result = tf.tensordot(self.neg_w, neg_weights, 1) + tf.tensordot(self.pos_w, pos_weights, 1)
        return result

    def apply_constraint(self, neg_cst, pos_cst):
        weights = _apply_dense(self.pos_w, pos_cst.weights) + _apply_dense((self.neg_w, neg_cst.weights)
        return DenseConstraint(weights)


class OneToOneConstraint:

    def __init__(self, weights):
        assert len(weights.shape) == 1
        self.pos_w = tf.maximum(weights, 0)
        self.neg_w = tf.minimum(weights, 0)
        self.weights = weights

    def apply_weights(self, neg_weights, pos_weights):
        return self.pos_w * pos_weights + self.neg_w * neg_weights

    def apply_constraint(self, neg_cst, pos_cst):
        weights = _apply_dense(self.pos_w, pos_cst.weights) + _apply_dense((self.neg_w, neg_cst.weights)
        return DenseConstraint(weights)

"""

class DeepPolyElement:
    def __init__(self, lower_cst, upper_cst, lower_bias, upper_bias):
        self.lower_cst = lower_cst
        self.upper_cst = upper_cst
        self.lower_bias = lower_bias
        self.upper_bias = upper_bias
