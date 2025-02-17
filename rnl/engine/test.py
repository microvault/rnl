import timeit
import numpy as np
from shapely.geometry import Polygon, Point
from numba import njit, prange

# ----- Shapely -----
# Definindo o polígono com buracos
exterior_list = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
interiors_list = [[(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)]]
poly_shapely = Polygon(exterior_list, holes=interiors_list)

# Gerando 1 milhão de pontos aleatórios no intervalo [0,4]x[0,4]
n_points = 1_000_00
points = np.random.rand(n_points, 2) * 4.0

def shapely_batch():
    return [poly_shapely.contains(Point(x, y)) for x, y in points]

# ----- Numba (paralelo) -----
@njit(inline='always')
def ray_cast(x, y, poly):
    inside = False
    n = poly.shape[0]
    j = n - 1
    for i in range(n):
        xi = poly[i, 0]
        yi = poly[i, 1]
        xj = poly[j, 0]
        yj = poly[j, 1]
        if (yi > y) != (yj > y):
            intersect_x = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < intersect_x:
                inside = not inside
        j = i
    return inside

@njit
def contains_point(x, y, exterior, holes):
    # Se não está no exterior, já retorna False
    if not ray_cast(x, y, exterior):
        return False
    # Se estiver em algum buraco, retorna False
    for h in holes:
        if ray_cast(x, y, h):
            return False
    return True

@njit(parallel=True)
def batch_contains_points(points, exterior, holes):
    n_points = points.shape[0]
    result = np.empty(n_points, dtype=np.bool_)
    for i in prange(n_points):
        result[i] = contains_point(points[i, 0], points[i, 1], exterior, holes)
    return result

# Preparando os dados para Numba
exterior_np = np.array([[0.0, 0.0],
                        [4.0, 0.0],
                        [4.0, 4.0],
                        [0.0, 4.0]])
hole_np = np.array([[1.0, 1.0],
                    [3.0, 1.0],
                    [3.0, 3.0],
                    [1.0, 3.0]])
holes_np = (hole_np,)

# "Warm-up" para compilar as funções
_ = batch_contains_points(points, exterior_np, holes_np)
_ = shapely_batch()

# Medindo o tempo com timeit para 10 execuções
time_shapely = timeit.timeit(shapely_batch, number=10)
time_numba   = timeit.timeit(lambda: batch_contains_points(points, exterior_np, holes_np), number=10)

print("Tempo médio com Shapely:", time_shapely / 10)
print("Tempo médio com Numba (paralelo):", time_numba / 10)
