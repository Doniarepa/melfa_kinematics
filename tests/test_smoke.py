
import sys
from pathlib import Path

import numpy as np

# Permitir importar el paquete desde src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from melfa_kinematics.transforms import Rx, Ry, Rz, Trans


def test_rotaciones_identidad():
    identidad = np.eye(4)

    assert np.allclose(Rx(0), identidad)
    assert np.allclose(Ry(0), identidad)
    assert np.allclose(Rz(0), identidad)


def test_traslaciones():
    assert np.allclose(Trans(0, 0, 0), np.eye(4))

    esperado = np.eye(4)
    esperado[:3, 3] = [1, 2, 3]

    assert np.allclose(Trans(1, 2, 3), esperado)


def test_rotacion_z_90_grados():
    esperado = np.array([
        [0, -1, 0, 0],
        [1,  0, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1]
    ])

    assert np.allclose(Rz(np.pi / 2), esperado)


if __name__ == "__main__":
    test_rotaciones_identidad()
    test_traslaciones()
    test_rotacion_z_90_grados()

    print("Todas las pruebas MELFA pasaron correctamente.")
