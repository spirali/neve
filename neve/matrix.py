import tensorflow as tf


class Matrix:
    """
    Helper class for matrices for optimizations
    over full matrices vs diagonal matrices
    """

    pass


class FullMatrix(Matrix):

    def __init__(self, values):
        self.values = values

    def __matmul__(self, other):
        if isinstance(other, FullMatrix):
            return FullMatrix(self.values @ other.values)
        if isinstance(other, DiagonalMatrix):
            return FullMatrix(self.values * other.values)
        raise Exception("Invalid type")

    def __add__(self, other):
        if isinstance(other, FullMatrix):
            return FullMatrix(self.values + other.values)
        if isinstance(other, DiagonalMatrix):
            raise NotImplementedError()
        raise Exception("Invalid type")

    def tensordot(self, column):
        return tf.tensordot(self.values, column, 1)

    def positive(self):
        return FullMatrix(tf.maximum(self.values, 0))

    def negative(self):
        return FullMatrix(tf.minimum(self.values, 0))


class DiagonalMatrix(Matrix):

    def __init__(self, values):
        self.values = values

    def __matmul__(self, other):
        if isinstance(other, FullMatrix):
            return FullMatrix(other.values * tf.reshape(self.values, (-1, 1)))
        if isinstance(other, DiagonalMatrix):
            return DiagonalMatrix(self.values * other.values)
        raise Exception("Invalid type")

    def __add__(self, other):
        if isinstance(other, FullMatrix):
            raise NotImplementedError()
        if isinstance(other, DiagonalMatrix):
            return DiagonalMatrix(self.values + other.values)
        raise Exception("Invalid type")

    def tensordot(self, column):
        return self.values * column

    def positive(self):
        return DiagonalMatrix(tf.maximum(self.values, 0))

    def negative(self):
        return DiagonalMatrix(tf.minimum(self.values, 0))
