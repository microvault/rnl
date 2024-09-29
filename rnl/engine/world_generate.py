import numpy as np

# import numpy.random as npr
from numba import njit


class GenerateWorld:
    def __init__(self):
        pass

    def generate_maze(
        self,
        map_size: int,
        decimation: float = 0.0,
        min_blocks: int = 10,
        num_cells_togo: int = 100,
    ) -> np.ndarray:
        return generate_maze(map_size, decimation, min_blocks, num_cells_togo)


# @njit(parallel=True)
# def generate_maze(
#     map_size: int,
#     decimation: float = 0.0,
#     min_blocks: int = 10,
#     num_cells_togo: int = 150,
# ) -> np.ndarray:
#     """
#     Generates a maze using Kruskal's algorithm.

#     Parameters:
#     map_size (int): The size of the maze.
#     decimation (float): The probability of removing blocks.
#     min_blocks (int): The minimum number of blocks to keep.
#     num_cells_togo (int): The number of cells to remove.

#     Returns:
#     np.ndarray: The generated maze represented as a binary array.
#     """
#     m = (map_size - 1) // 2
#     n = (map_size - 1) // 2
#     maze = np.ones((map_size, map_size))
#     maze = np.ones((map_size, map_size))
#     for i in range(m):
#         for j in range(n):
#             maze[2 * i + 1, 2 * j + 1] = 0
#     m = m - 1
#     L = np.arange(n + 1)
#     R = np.arange(n)
#     L[n] = n - 1

#     while m > 0:
#         for i in range(n):
#             j = L[i + 1]
#             if i != j and npr.randint(3) != 0:
#                 R[j] = R[i]
#                 L[R[j]] = j
#                 R[i] = i + 1
#                 L[R[i]] = i
#                 maze[2 * (n - m) - 1, 2 * i + 2] = 0
#             if i != L[i] and npr.randint(3) != 0:
#                 L[R[i]] = L[i]
#                 R[L[i]] = R[i]
#                 L[i] = i
#                 R[i] = i
#             else:
#                 maze[2 * (n - m), 2 * i + 1] = 0

#         m -= 1

#     for i in range(n):
#         j = L[i + 1]
#         if i != j and (i == L[i] or npr.randint(3) != 0):
#             R[j] = R[i]
#             L[R[j]] = j
#             R[i] = i + 1
#             L[R[i]] = i
#             maze[2 * (n - m) - 1, 2 * i + 2] = 0

#         L[R[i]] = L[i]
#         R[L[i]] = R[i]
#         L[i] = i
#         R[i] = i

#     # ----- Generate Map -----
#     index_ones = np.arange(map_size * map_size)[maze.flatten() == 1]

#     if index_ones.size < min_blocks:
#         raise ValueError("Minimum number of blocks cannot be placed.")

#     reserve = min(index_ones.size, min_blocks)
#     num_cells_togo = min(num_cells_togo, index_ones.size - reserve)

#     for _ in range(num_cells_togo):
#         blocks_remove = np.random.randint(0, map_size**2)
#         row_index = blocks_remove // map_size
#         col_index = blocks_remove % map_size
#         maze[row_index, col_index] = 0

#     return maze


@njit(parallel=True)
def generate_maze(
    map_size: int,
    decimation: float = 0.0,
    min_blocks: int = 1,
    num_cells_togo: int = 150,
) -> np.ndarray:
    """
    Generates a maze using Kruskal's algorithm without randomness.

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

        m -= 1

    for i in range(n):
        j = L[i + 1]
        if i != j and ((i + j) % 2 == 0):
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

    for i in range(num_cells_togo):
        blocks_remove = (i * 7) % (map_size**2)
        row_index = blocks_remove // map_size
        col_index = blocks_remove % map_size
        maze[row_index, col_index] = 0

    return maze
