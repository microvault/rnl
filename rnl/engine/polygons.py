import numpy as np
from numba import njit, prange, typed


def compute_polygon_diameter(poly) -> float:
    """
    Calcula a máxima distância (diâmetro) entre dois pontos do polígono.
    Usa o convex hull do polígono para reduzir o número de pontos.
    """
    # Obtém o convex hull do polígono
    convex = poly.convex_hull
    pts = np.array(convex.exterior.coords)

    max_dist = 0.0
    n = len(pts)

    # Busca dupla para calcular todas as distâncias entre os vértices
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(pts[i] - pts[j])
            if dist > max_dist:
                max_dist = dist

    return max_dist


@njit(inline="always")
def ray_cast(x: float, y: float, poly: np.ndarray) -> bool:
    """
    Verifica se o ponto (x, y) está dentro do polígono usando ray casting.
    """
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
def contains_point(x: float, y: float, exterior: np.ndarray, holes: typed.List) -> bool:
    """
    Retorna True se o ponto (x, y) estiver dentro do contorno exterior e fora dos buracos.
    """
    if not ray_cast(x, y, exterior):
        return False
    for h in holes:
        if ray_cast(x, y, h):
            return False
    return True


@njit(parallel=True)
def batch_contains_points(
    points: np.ndarray, exterior: np.ndarray, holes: typed.List
) -> np.ndarray:
    """
    Verifica em batch se os pontos estão contidos na região definida.
    """
    n_points = points.shape[0]
    result = np.empty(n_points, dtype=np.bool_)
    for i in prange(n_points):
        result[i] = contains_point(points[i, 0], points[i, 1], exterior, holes)
    return result


@njit(fastmath=True)
def interpolate_point(
    x0: float, y0: float, x1: float, y1: float, v0: float, v1: float, level: float
) -> (float, float):
    """
    Interpola linearmente entre dois pontos para encontrar o cruzamento com 'level'.
    """
    t = (level - v0) / (v1 - v0) if v1 != v0 else 0.5
    return x0 + t * (x1 - x0), y0 + t * (y1 - y0)


@njit(fastmath=True)
def process_cell(i: int, j: int, arr: np.ndarray, level: float) -> typed.List:
    """
    Processa uma célula da grade e retorna os segmentos do contorno gerado.
    """
    segs = typed.List()
    a = arr[i, j]
    b = arr[i, j + 1]
    c = arr[i + 1, j + 1]
    d = arr[i + 1, j]

    idx = 0
    if a >= level:
        idx |= 1
    if b >= level:
        idx |= 2
    if c >= level:
        idx |= 4
    if d >= level:
        idx |= 8
    if idx == 0 or idx == 15:
        return segs

    x_top, y_top = interpolate_point(j, i, j + 1, i, a, b, level)
    x_right, y_right = interpolate_point(j + 1, i, j + 1, i + 1, b, c, level)
    x_bottom, y_bottom = interpolate_point(j, i + 1, j + 1, i + 1, d, c, level)
    x_left, y_left = interpolate_point(j, i, j, i + 1, a, d, level)

    seg = np.empty((2, 2), dtype=np.float64)

    if idx == 1:
        seg[0, :] = np.array([x_left, y_left])
        seg[1, :] = np.array([x_top, y_top])
        segs.append(seg)
    elif idx == 2:
        seg[0, :] = np.array([x_top, y_top])
        seg[1, :] = np.array([x_right, y_right])
        segs.append(seg)
    elif idx == 3:
        seg[0, :] = np.array([x_left, y_left])
        seg[1, :] = np.array([x_right, y_right])
        segs.append(seg)
    elif idx == 4:
        seg[0, :] = np.array([x_right, y_right])
        seg[1, :] = np.array([x_bottom, y_bottom])
        segs.append(seg)
    elif idx == 5:
        seg1 = np.empty((2, 2), dtype=np.float64)
        seg1[0, :] = np.array([x_left, y_left])
        seg1[1, :] = np.array([x_top, y_top])
        segs.append(seg1)
        seg2 = np.empty((2, 2), dtype=np.float64)
        seg2[0, :] = np.array([x_right, y_right])
        seg2[1, :] = np.array([x_bottom, y_bottom])
        segs.append(seg2)
    elif idx == 6:
        seg[0, :] = np.array([x_top, y_top])
        seg[1, :] = np.array([x_bottom, y_bottom])
        segs.append(seg)
    elif idx == 7:
        seg[0, :] = np.array([x_left, y_left])
        seg[1, :] = np.array([x_bottom, y_bottom])
        segs.append(seg)
    elif idx == 8:
        seg[0, :] = np.array([x_bottom, y_bottom])
        seg[1, :] = np.array([x_left, y_left])
        segs.append(seg)
    elif idx == 9:
        seg[0, :] = np.array([x_top, y_top])
        seg[1, :] = np.array([x_bottom, y_bottom])
        segs.append(seg)
    elif idx == 10:
        seg1 = np.empty((2, 2), dtype=np.float64)
        seg1[0, :] = np.array([x_top, y_top])
        seg1[1, :] = np.array([x_right, y_right])
        segs.append(seg1)
        seg2 = np.empty((2, 2), dtype=np.float64)
        seg2[0, :] = np.array([x_left, y_left])
        seg2[1, :] = np.array([x_bottom, y_bottom])
        segs.append(seg2)
    elif idx == 11:
        seg[0, :] = np.array([x_right, y_right])
        seg[1, :] = np.array([x_bottom, y_bottom])
        segs.append(seg)
    elif idx == 12:
        seg[0, :] = np.array([x_right, y_right])
        seg[1, :] = np.array([x_left, y_left])
        segs.append(seg)
    elif idx == 13:
        seg[0, :] = np.array([x_top, y_top])
        seg[1, :] = np.array([x_right, y_right])
        segs.append(seg)
    elif idx == 14:
        seg[0, :] = np.array([x_left, y_left])
        seg[1, :] = np.array([x_top, y_top])
        segs.append(seg)

    return segs


@njit(fastmath=True)
def compute_segments(arr: np.ndarray, level: float) -> typed.List:
    """
    Computa os segmentos do contorno para toda a matriz usando marching squares.
    """
    segments = typed.List()
    rows, cols = arr.shape
    for i in range(rows - 1):
        for j in range(cols - 1):
            cell_segs = process_cell(i, j, arr, level)
            for seg in cell_segs:
                segments.append(seg)
    return segments


@njit
def list_insert_front(lst: typed.List, item: np.ndarray) -> typed.List:
    """
    Insere um item no início de uma typed.List.
    """
    new_lst = typed.List()
    new_lst.append(item)
    for i in range(len(lst)):
        new_lst.append(lst[i])
    return new_lst


@njit
def try_extend_contour(
    contour: typed.List,
    segments: typed.List,
    used: np.ndarray,
    tol2: float,
    at_start: bool,
) -> (typed.List, bool):
    """
    Tenta estender o contorno adicionando segmentos conectados no início ou no final.
    """
    target = contour[0] if at_start else contour[-1]
    for j in range(len(segments)):
        if not used[j]:
            s = segments[j]
            dx = s[0, 0] - target[0]
            dy = s[0, 1] - target[1]
            if dx * dx + dy * dy < tol2:
                if at_start:
                    contour = list_insert_front(contour, s[1])
                else:
                    contour.append(s[1])
                used[j] = True
                return contour, True
            dx = s[1, 0] - target[0]
            dy = s[1, 1] - target[1]
            if dx * dx + dy * dy < tol2:
                if at_start:
                    contour = list_insert_front(contour, s[0])
                else:
                    contour.append(s[0])
                used[j] = True
                return contour, True
    return contour, False


@njit(fastmath=True)
def build_contours_from_segments(segments: typed.List, tol: float = 1e-6) -> typed.List:
    """
    Constrói contornos conectando os segmentos adjacentes.
    """
    nseg = len(segments)
    used = np.zeros(nseg, dtype=np.bool_)
    contours = typed.List()
    tol2 = tol * tol
    for i in range(nseg):
        if used[i]:
            continue
        used[i] = True
        contour = typed.List()
        for k in range(segments[i].shape[0]):
            contour.append(segments[i][k])
        extended = True
        while extended:
            extended = False
            contour, ext_end = try_extend_contour(contour, segments, used, tol2, False)
            if ext_end:
                extended = True
            contour, ext_start = try_extend_contour(contour, segments, used, tol2, True)
            if ext_start:
                extended = True
        contours.append(contour)
    return contours


def find_contour(arr: np.ndarray, level: float = 0.5) -> typed.List:
    """
    Encontra os contornos de uma matriz de valores usando marching squares.
    """
    segments = compute_segments(arr, level)
    contours = build_contours_from_segments(segments)
    return contours


def process(contours: typed.List) -> list:
    """
    Processa os contornos, fechando-os se necessário e descartando contornos inválidos.
    """
    processed = []
    for contour in contours:
        pts = [np.array(pt) for pt in contour]
        if len(pts) < 3:
            continue
        if not np.allclose(pts[0], pts[-1]):
            pts.append(pts[0])
        if len(pts) < 4:
            continue
        processed.append(np.array(pts))
    return processed
