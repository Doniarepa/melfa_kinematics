
import numpy as np


def Rx(theta):
    """Rotación homogénea sobre X.

    theta: ángulo en radianes.
    """
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [1, 0,  0, 0],
        [0, c, -s, 0],
        [0, s,  c, 0],
        [0, 0,  0, 1]
    ], dtype=float)


def Ry(theta):
    """Rotación homogénea sobre Y.

    theta: ángulo en radianes.
    """
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1]
    ], dtype=float)


def Rz(theta):
    """Rotación homogénea sobre Z.

    theta: ángulo en radianes.
    """
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1]
    ], dtype=float)


def Trans(x, y, z):
    """Traslación homogénea en los ejes X, Y y Z."""

    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ], dtype=float)

