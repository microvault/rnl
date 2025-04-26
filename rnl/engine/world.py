import numpy as np
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
        no_mut: bool,
        porcentage_obstacle: float,
    ) -> np.ndarray:
        return generate_maze(
            map_size, decimation, min_blocks, no_mut, porcentage_obstacle
        )


@njit(parallel=True)
def generate_maze(
    map_size: int,
    decimation: float,
    min_blocks: int,
    no_mut: bool,
    porcentage_obstacle: float,
) -> np.ndarray:
    """
    Gera um labirinto usando o algoritmo de Kruskal.

    Parâmetros:
    map_size (int): Tamanho do labirinto.
    decimation (float): Probabilidade de remover blocos (não utilizado na lógica atual).
    min_blocks (int): Número mínimo de blocos (obstáculos) a manter.
    no_mut (bool): Flag para controlar o método de mutação.
    porcentage_obstaculo (float): Porcentagem desejada de obstáculos (0 a 100).

    Retorna:
    np.ndarray: Labirinto gerado representado como uma matriz binária.
    """
    m = (map_size - 1) // 2
    n = (map_size - 1) // 2
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
                if i != j and np.random.randint(0, 3) != 0:
                    R[j] = R[i]
                    L[R[j]] = j
                    R[i] = i + 1
                    L[R[i]] = i
                    maze[2 * (n - m) - 1, 2 * i + 2] = 0
                if i != L[i] and np.random.randint(0, 3) != 0:
                    L[R[i]] = L[i]
                    R[L[i]] = R[i]
                    L[i] = i
                    R[i] = i
                else:
                    maze[2 * (n - m), 2 * i + 1] = 0
        m -= 1

    for i in range(n):
        j = L[i + 1]
        if i != j and (i == L[i] or np.random.randint(0, 3) != 0):
            R[j] = R[i]
            L[R[j]] = j
            R[i] = i + 1
            L[R[i]] = i
            maze[2 * (n - m) - 1, 2 * i + 2] = 0

        L[R[i]] = L[i]
        R[L[i]] = R[i]
        L[i] = i
        R[i] = i

    total_cells = map_size * map_size
    desired_obstacles = int(round(porcentage_obstacle / 100.0 * total_cells))
    if desired_obstacles < min_blocks:
        desired_obstacles = min_blocks

    current_obstacles = np.sum(maze == 1)
    while current_obstacles > desired_obstacles:
        idx = np.random.randint(0, total_cells)
        row_index = idx // map_size
        col_index = idx % map_size
        if maze[row_index, col_index] == 1:
            maze[row_index, col_index] = 0
            current_obstacles -= 1

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

    minx_r, miny_r, maxx_r, maxy_r = safe_poly_robot.bounds
    minx_g, miny_g, maxx_g, maxy_g = safe_poly_goal.bounds

    ext_robot_arr = np.array(safe_poly_robot.exterior.coords, dtype=np.float64)
    holes_robot_arr = [
        np.array(i.coords, dtype=np.float64) for i in safe_poly_robot.interiors
    ]

    ext_goal_arr = np.array(safe_poly_goal.exterior.coords, dtype=np.float64)
    holes_goal_arr = [
        np.array(i.coords, dtype=np.float64) for i in safe_poly_goal.interiors
    ]

    if len(holes_robot_arr) == 0:
        holes_robot_arr.append(np.zeros((0, 2), dtype=np.float64))
    if len(holes_goal_arr) == 0:
        holes_goal_arr.append(np.zeros((0, 2), dtype=np.float64))

    holes_robot_nb = List.empty_list(types.float64[:, :])
    for arr in holes_robot_arr:
        holes_robot_nb.append(arr)

    holes_goal_nb = List.empty_list(types.float64[:, :])
    for arr in holes_goal_arr:
        holes_goal_nb.append(arr)

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
            return (robo_x, robo_y), (goal_x, goal_y)

    raise ValueError("Falha ao gerar posições válidas para robô e objetivo.")
