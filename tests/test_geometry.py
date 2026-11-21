import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from melfa_kinematics.geometry import transform_point
from melfa_kinematics.transforms import Rz, Trans


def test_identidad():
    point = (1, -2, 3)

    assert np.allclose(transform_point(np.eye(4), point), point)


def test_traslacion():
    assert np.allclose(transform_point(Trans(1, 2, 3), (4, 5, 6)), (5, 7, 9))


def test_rotacion_y_traslacion():
    T = Trans(1, 2, 3) @ Rz(np.pi / 2)

    assert np.allclose(transform_point(T, (1, 0, 0)), (1, 3, 3))


def test_matriz_con_forma_invalida():
    with np.testing.assert_raises(ValueError):
        transform_point(np.eye(3), (1, 2, 3))


def test_punto_con_forma_invalida():
    with np.testing.assert_raises(ValueError):
        transform_point(np.eye(4), (1, 2))


if __name__ == "__main__":
    test_identidad()
    test_traslacion()
    test_rotacion_y_traslacion()
    test_matriz_con_forma_invalida()
    test_punto_con_forma_invalida()

    print("Todas las pruebas de geometría pasaron correctamente.")
