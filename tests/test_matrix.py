import tensorflow as tf
import numpy as np
from neve.matrix import FullMatrix, DiagonalMatrix


def test_add():
    c1 = FullMatrix(tf.constant([[1, 2, 3], [3, 4, 5]]))
    c2 = FullMatrix(tf.constant([[10, 20, 30], [30, 0, 0]]))

    r = c1 + c2
    assert isinstance(r, FullMatrix)
    assert (r.values.numpy() == np.array([[11, 22, 33], [33, 4, 5]])).all()

    c3 = np.array([1, 2, 3])
    c4 = np.array([4, 2, 0])

    m3 = DiagonalMatrix(tf.constant(c3))
    m4 = DiagonalMatrix(tf.constant(c4))

    r = m3 + m4
    assert list(r.values.numpy()) == [5, 4, 3]


def test_matmul():
    c1 = np.array([[1, 2, 3], [3, 4, 5], [0, 0, 3]])
    c2 = np.array([[2, 3], [3, 4], [1, -2]])

    m1 = FullMatrix(tf.constant(c1))
    m2 = FullMatrix(tf.constant(c2))

    r = m1 @ m2

    assert isinstance(r, FullMatrix)
    assert (np.array_equal(r.values.numpy(), (c1 @ c2)))

    c3 = np.array([-1, 2, 3])
    c4 = np.array([4, 2, 1])

    m3 = DiagonalMatrix(tf.constant(c3))
    m4 = DiagonalMatrix(tf.constant(c4))

    r = m3 @ m4
    assert list(r.values.numpy()) == [-4, 4, 3]

    r = m1 @ m3
    assert isinstance(r, FullMatrix)
    assert np.array_equal(r.values.numpy(), (c1 @ np.diag(c3)))


    r = m3 @ m1
    assert isinstance(r, FullMatrix)
    assert np.array_equal(r.values.numpy(), (np.diag(c3) @ c1))


def test_tensordot():
    c1 = np.array([[1, 2, 3], [3, 4, 5], [0, 0, 3]])
    m1 = FullMatrix(tf.constant(c1))
    c3 = np.array([-1, 2, 3])
    m3 = DiagonalMatrix(tf.constant(c3))

    r = m1.tensordot(tf.constant(np.array([2, 3, 4])))
    assert list(r.numpy()) == [20, 38, 12]

    r = m3.tensordot(tf.constant(np.array([2, 3, 4])))
    assert list(r.numpy()) == [-2, 6, 12]