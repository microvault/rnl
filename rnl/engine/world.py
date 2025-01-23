import numpy as np
import numpy.random as npr
from numba import njit, types
from numba.typed import List


class GenerateWorld:
    def __init__(self):
        pass

    def generate_maze(
        self,
        map_size: int,
        decimation: float,
        min_blocks: int,
        num_cells_togo: int,
        no_mut: bool,
    ) -> np.ndarray:
        return generate_maze(map_size, decimation, min_blocks, num_cells_togo, no_mut)


@njit(parallel=True)
def generate_maze(
    map_size: int,
    decimation: float,
    min_blocks: int,
    num_cells_togo: int,
    no_mut: bool,
) -> np.ndarray:
    """
    Generates a maze using Kruskal's algorithm.

    Parameters:
    map_size (int): The size of the maze.
    decimation (float): The probability of removing blocks.
    min_blocks (int): The minimum number of blocks to keep.
    num_cells_togo (int): The number of cells to remove.

    Returns:
    np.ndarray: The generated maze represented as a binary array.
    """
    m = (map_size - 1) // 2
    n = (map_size - 1) // 2
    maze = np.ones((map_size, map_size))
    maze = np.ones((map_size, map_size))
    for i in range(m):
        for j in range(n):
            maze[2 * i + 1, 2 * j + 1] = 0
    m = m - 1
    L = np.arange(n + 1)
    R = np.arange(n)
    L[n] = n - 1

    while m > 0:
        for i in range(n):
            j = L[i + 1]
            if no_mut:
                if i != j and (i + j) % 2 == 0:
                    R[j] = R[i]
                    L[R[j]] = j
                    R[i] = i + 1
                    L[R[i]] = i
                    maze[2 * (n - m) - 1, 2 * i + 2] = 0

                if i != L[i] and (i + j) % 2 == 0:
                    L[R[i]] = L[i]
                    R[L[i]] = R[i]
                    L[i] = i
                    R[i] = i
                else:
                    maze[2 * (n - m), 2 * i + 1] = 0
            else:
                if i != j and npr.randint(3) != 0:
                    R[j] = R[i]
                    L[R[j]] = j
                    R[i] = i + 1
                    L[R[i]] = i
                    maze[2 * (n - m) - 1, 2 * i + 2] = 0
                if i != L[i] and npr.randint(3) != 0:
                    L[R[i]] = L[i]
                    R[L[i]] = R[i]
                    L[i] = i
                    R[i] = i
                else:
                    maze[2 * (n - m), 2 * i + 1] = 0
        m -= 1

    for i in range(n):
        j = L[i + 1]
        if i != j and (i == L[i] or npr.randint(3) != 0):
            R[j] = R[i]
            L[R[j]] = j
            R[i] = i + 1
            L[R[i]] = i
            maze[2 * (n - m) - 1, 2 * i + 2] = 0

        L[R[i]] = L[i]
        R[L[i]] = R[i]
        L[i] = i
        R[i] = i

    # ----- Generate Map -----
    index_ones = np.arange(map_size * map_size)[maze.flatten() == 1]

    if index_ones.size < min_blocks:
        raise ValueError("Minimum number of blocks cannot be placed.")

    reserve = min(index_ones.size, min_blocks)
    num_cells_togo = min(num_cells_togo, index_ones.size - reserve)

    for _ in range(num_cells_togo):
        blocks_remove = np.random.randint(0, map_size**2)
        row_index = blocks_remove // map_size
        col_index = blocks_remove % map_size
        maze[row_index, col_index] = 0

    return maze


@njit
def _point_in_ring(px, py, ring):
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        # Ray casting
        if ((y1 > py) != (y2 > py)) and (
            px < (x2 - x1) * (py - y1) / (y2 - y1 + 1e-15) + x1
        ):
            inside = not inside
    return inside


@njit
def _point_in_polygon(px, py, exterior, holes):
    # Verifica se está dentro do contorno exterior
    if not _point_in_ring(px, py, exterior):
        return False
    # Se encontrar dentro de um furo, retorna False
    for hole in holes:
        if _point_in_ring(px, py, hole):
            return False
    return True


@njit
def random_point_in_poly(bounds, ext, holes, max_tries_local):
    minx, miny, maxx, maxy = bounds
    for _ in range(max_tries_local):
        rx = np.random.uniform(minx, maxx)
        ry = np.random.uniform(miny, maxy)
        if _point_in_polygon(rx, ry, ext, holes):
            return rx, ry
    return -1.0, -1.0


def spawn_robot_and_goal_with_maze(
    poly,
    robot_clearance=3.0,
    goal_clearance=3.0,
    min_robot_goal_dist=2.0,
    max_tries=1000,
):
    # Faz buffer negativo para clearance
    safe_poly_robot = poly.buffer(-robot_clearance)
    safe_poly_goal = poly.buffer(-goal_clearance)

    if safe_poly_robot.is_empty or safe_poly_goal.is_empty:
        raise ValueError("Clearance muito grande. Polígono invalido.")

    def to_single_polygon(shp):
        if shp.geom_type == "Polygon":
            return shp
        elif shp.geom_type == "MultiPolygon":
            largest = max(shp.geoms, key=lambda g: g.area)
            return largest
        else:
            raise ValueError("Geometria não é Polygon nem MultiPolygon.")

    safe_poly_robot = to_single_polygon(safe_poly_robot)
    safe_poly_goal = to_single_polygon(safe_poly_goal)

    # Extrai bounds
    minx_r, miny_r, maxx_r, maxy_r = safe_poly_robot.bounds
    minx_g, miny_g, maxx_g, maxy_g = safe_poly_goal.bounds

    # Extrai exterior e buracos como np.ndarray de float64
    ext_robot_arr = np.array(safe_poly_robot.exterior.coords, dtype=np.float64)
    holes_robot_arr = [
        np.array(i.coords, dtype=np.float64) for i in safe_poly_robot.interiors
    ]

    ext_goal_arr = np.array(safe_poly_goal.exterior.coords, dtype=np.float64)
    holes_goal_arr = [
        np.array(i.coords, dtype=np.float64) for i in safe_poly_goal.interiors
    ]

    # Se precisar, coloque array vazio se não existirem furos
    if len(holes_robot_arr) == 0:
        holes_robot_arr.append(np.zeros((0, 2), dtype=np.float64))
    if len(holes_goal_arr) == 0:
        holes_goal_arr.append(np.zeros((0, 2), dtype=np.float64))

    # Cria typed.List
    holes_robot_nb = List.empty_list(types.float64[:, :])
    for arr in holes_robot_arr:
        holes_robot_nb.append(arr)

    holes_goal_nb = List.empty_list(types.float64[:, :])
    for arr in holes_goal_arr:
        holes_goal_nb.append(arr)

    # Agora chamamos a função JIT
    for _ in range(max_tries):
        robo_x, robo_y = random_point_in_poly(
            (minx_r, miny_r, maxx_r, maxy_r), ext_robot_arr, holes_robot_nb, max_tries
        )
        if robo_x < 0:  # Falhou pra achar posicao do robo
            continue

        goal_x, goal_y = random_point_in_poly(
            (minx_g, miny_g, maxx_g, maxy_g), ext_goal_arr, holes_goal_nb, max_tries
        )
        if goal_x < 0:  # Falhou pra achar posicao da meta
            continue

        dx = robo_x - goal_x
        dy = robo_y - goal_y
        dist = (dx * dx + dy * dy) ** 0.5

        if dist >= min_robot_goal_dist:
            # Achou posições válidas
            return (robo_x, robo_y), (goal_x, goal_y)

    raise ValueError("Falha ao gerar posições válidas para robô e objetivo.")
