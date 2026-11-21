import numpy as np


def transform_point(T, point):
    """Aplica una transformación homogénea afín 4×4 a un punto de forma (3,).

    Devuelve las tres coordenadas cartesianas como un array de NumPy.
    """
    matrix = np.asarray(T, dtype=float)
    coordinates = np.asarray(point, dtype=float)

    if matrix.shape != (4, 4):
        raise ValueError("T debe tener forma (4, 4).")
    if coordinates.shape != (3,):
        raise ValueError("point debe tener forma (3,).")

    homogeneous_point = np.append(coordinates, 1.0)
    return (matrix @ homogeneous_point)[:3]
