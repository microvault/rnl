from functools import lru_cache
from typing import List, Tuple, Optional
import numpy as np
from numba import njit
# from numba import njit
from typing import List, Tuple
import numpy as np
from dataclasses import dataclass
from rtree import index

class SpatialIndex:
    def __init__(self, segments: List[Tuple[float, float, float, float]]):
        """
        Inicializa o índice espacial com os segmentos fornecidos.

        Parameters:
        - segments: Lista de segmentos do mapa, cada um definido por (x1, y1, x2, y2).
        """
        p = index.Property()
        p.dimension = 2
        self.idx = index.Index(properties=p)
        for i, seg in enumerate(segments):
            x1, y1, x2, y2 = seg
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            self.idx.insert(i, (xmin, ymin, xmax, ymax))
        self.segments = segments

    def query(self, x: float, y: float, max_range: float) -> List[Tuple[float, float, float, float]]:
        """
        Consulta os segmentos que estão dentro do alcance máximo a partir da posição (x, y).

        Parameters:
        - x, y: Coordenadas do robô.
        - max_range: Alcance máximo para filtrar os segmentos.

        Returns:
        - Lista de segmentos filtrados.
        """
        return [
            self.segments[i]
            for i in self.idx.intersection((x - max_range, y - max_range, x + max_range, y + max_range))
        ]

class Collision:
    def __init__(self, spatial_index: Optional[SpatialIndex] = None):
        """
        Inicializa a detecção de colisão.

        Parameters:
        - segments: Lista de segmentos do mapa, cada um definido por (x1, y1, x2, y2).
        - spatial_index: (Opcional) Instância de SpatialIndex para otimização.
        """
        self.spatial_index = spatial_index

    def filter_segments(self, x: float, y: float, max_range: float, segments = None) -> List[Tuple[float, float, float, float]]:
        """
        Filtra os segmentos com base na posição do robô e no alcance máximo.

        Parameters:
        - x, y: Coordenadas do robô.
        - max_range: Alcance máximo para filtrar os segmentos.

        Returns:
        - Lista de segmentos filtrados.
        """
        if self.spatial_index is not None:
            segs_proximos = self.spatial_index.query(x, y, max_range)
            segs_filtrados = filter_list_segment(segs_proximos, x, y, max_range)
        else:
            segs_filtrados = filter_list_segment_map_generated(segments, x, y, max_range)

        return segs_filtrados

    def lidar_intersection(
        self,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        lidar_range: float,
        lidar_angles: np.ndarray,
        segments: List[Tuple[float, float, float, float]],
    ) -> List[Tuple[float, float]]:
        """
        Calcula as interseções dos raios do LiDAR com os segmentos filtrados.

        Parameters:
        - robot_x, robot_y: Coordenadas do robô.
        - robot_theta: Orientação do robô.
        - lidar_range: Alcance máximo do LiDAR.
        - lidar_angles: Ângulos dos raios do LiDAR.
        - segments: Segmentos filtrados para verificar interseções.

        Returns:
        - Lista de pontos de interseção.
        """
        if self.spatial_index is not None:
            return lidar_intersections(robot_x, robot_y, robot_theta, lidar_range, lidar_angles, segments)
        else:
            return lidar_intersections_map_generated(robot_x, robot_y, robot_theta, lidar_range, lidar_angles, segments)


    def lidar_measurement(
        self,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        lidar_range: float,
        lidar_angles: np.ndarray,
        segments: List[Tuple[float, float, float, float]],
    ) -> List[float]:
        """
        Calcula as medições dos raios do LiDAR com os segmentos filtrados.

        Parameters:
        - robot_x, robot_y: Coordenadas do robô.
        - robot_theta: Orientação do robô.
        - lidar_range: Alcance máximo do LiDAR.
        - lidar_angles: Ângulos dos raios do LiDAR.
        - segments: Segmentos filtrados para verificar interseções.

        Returns:
        - Lista de medições de distância.
        """
        if self.spatial_index is not None:
            return lidar_measurements(robot_x, robot_y, robot_theta, lidar_range, lidar_angles, segments)
        else:
            return lidar_measurements_map_generated(robot_x, robot_y, robot_theta, lidar_range, lidar_angles, segments)

@njit
def is_distance_within_range(distance: np.ndarray, lidar_range: float) -> bool:
    return np.all(distance <= lidar_range)

@njit
def calculate_distances(seg_ends: np.ndarray, center: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((seg_ends - center) ** 2, axis=1))

@njit
def cross_product_2d_vector(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcula o produto vetorial de dois vetores 2D.

    Parameters:
    - a, b: Vetores 2D.

    Returns:
    - Produto vetorial.
    """
    return a[0] * b[1] - a[1] * b[0]

@njit
def calculate_temp_vector(direction_vec: np.ndarray, r: np.ndarray, s: np.ndarray) -> Tuple[float, float]:
    cross_product = cross_product_2d_vector(r, s)

    if cross_product != 0:
        t = cross_product_2d_vector(direction_vec, s) / cross_product
        u = cross_product_2d_vector(direction_vec, r) / cross_product
    else:
        t = u = 1.0

    return t, u

@njit
def calculate_intersection_point_vec(p: np.ndarray, t: float, r: np.ndarray) -> np.ndarray:
    """
    Calcula o ponto de interseção entre dois segmentos de linha.

    Parameters:
    - p: Ponto de origem.
    - t: Parâmetro t.
    - r: Vetor de direção.

    Returns:
    - Ponto de interseção.
    """
    return p + t * r

@njit
def calculate_distance_to_intersection(inter_point: np.ndarray, p: np.ndarray) -> float:
    """
    Calcula a distância do ponto de origem até o ponto de interseção.

    Parameters:
    - inter_point: Ponto de interseção.
    - p: Ponto de origem.

    Returns:
    - Distância calculada.
    """
    return np.sqrt(np.sum((inter_point - p) ** 2))

@njit
def filter_list_segment_map_generated(segs: List, x: float, y: float, max_range: float) -> List:
    """
    Filters line segments based on proximity to a given point.

    Parameters:
    segs (list): List of line segments.
    x (float): X-coordinate of the point.
    y (float): Y-coordinate of the point.
    max_range (int): Maximum range for filtering segments.

    Returns:
    List: List of line segments filtered based on proximity to the given point.
    """
    segments_trans = [
        [np.array(segments[:2]), np.array(segments[2:])] for segments in segs
    ]

    segments_inside = []
    region_center = np.array([x, y])

    for segment in segments_trans:
        x1, y1 = segment[0]
        x2, y2 = segment[1]
        segment_endpoints = np.array([[x1, y1], [x2, y2]])

        distances = calculate_distances(segment_endpoints, region_center)

        if is_distance_within_range(distances, max_range):
            segments_inside.append(segment)

    return segments_inside


@njit
def filter_list_segment(segs: List[Tuple[float, float, float, float]], x: float, y: float, max_range: float) -> List:
    """
    Filtra segmentos com base na proximidade ao ponto (x, y).

    Parameters:
    - segs: Lista de segmentos (x1, y1, x2, y2).
    - x, y: Coordenadas do ponto central.
    - max_range: Alcance máximo para filtrar os segmentos.

    Returns:
    - Lista de segmentos filtrados.
    """
    segments_inside = []
    region_center = np.array([x, y])

    for (x1, y1, x2, y2) in segs:
        seg_ends = np.array([[x1, y1], [x2, y2]])
        distances = calculate_distances(seg_ends, region_center)
        if is_distance_within_range(distances, max_range):
            segments_inside.append((x1, y1, x2, y2))

    return segments_inside

@njit
def connect_polygon_points(polygon: np.ndarray) -> List:
    num_points = len(polygon)
    segments = []
    for i in range(num_points):
        current_point = polygon[i]
        next_point = polygon[(i + 1) % num_points]
        segment = (current_point, next_point)
        segments.append(segment)
    return segments

@njit
def convert_to_line_segments(total_segments: List) -> List:
    line_segments = []
    for segment in total_segments:
        line_segments.append((segment[0][0], segment[0][1], segment[1][0], segment[1][1]))
    return line_segments

@njit
def extract_segment_from_polygon(stack: List) -> List:
    total_segments = []
    for polygon in stack:
        segments = connect_polygon_points(np.array(polygon))
        total_segments.extend(segments)
    return convert_to_line_segments(total_segments)

# Interseção de Segmentos com Polígonos
def intersect_segment_with_polygon(segment, poly: List[Tuple[float,float,float,float]]) -> List:
    inter = []

    # Converte o segmento a ser testado (o "raio" do LiDAR) em arrays
    (x1_seg, y1_seg), (x2_seg, y2_seg) = segment
    x1_seg, y1_seg = segment[0]
    x2_seg, y2_seg = segment[1]
    p = np.array([x1_seg, y1_seg])
    r = np.array([x2_seg, y2_seg]) - p

    for seg in poly:
        # Converte cada segmento do polígono
        px, py, qx, qy = seg
        q = np.array([px, py])
        s = np.array([qx, qy]) - q

        temp = cross_product_2d_vector(r, s)
        if temp != 0:
            direction_vec = q - p
            t, u = calculate_temp_vector(direction_vec, r, s)
            if 0 <= t <= 1 and 0 <= u <= 1:
                interception_point = calculate_intersection_point_vec(p, t, r)
                distances = calculate_distance_to_intersection(interception_point, p)
                inter.append([True, interception_point, distances])

    return inter

def intersect_segment_with_polygon_map_generated(segment: list, poly: list) -> List:
    """
    Intersects a line segment with a polygon.

    Parameters:
    segment (list): Line segment represented by two points.
    poly (list): List of line segments representing a polygon.

    Returns:
    List: List of intersection points with p, t, r
    """

    # (q+s)  (p+r)
    #   *     *
    #    \   /
    #     \ /
    #      *  temp
    #     / \
    #    /   \
    #   /     \
    #  *       *
    # (p)      (q)

    inter = []

    for segments in poly:
        p = segment[0]
        r = segment[1] - segment[0]
        q = segments[0]
        s = segments[1] - segments[0]

        temp = cross_product_2d_vector(r, s)

        # If temp is 0, the lines are parallel (//)
        if temp != 0:
            direction_vec = q - p

            t, u = calculate_temp_vector(direction_vec, r, s)

            # If t and u are between 0 and 1
            if t >= 0 and t <= 1 and u >= 0 and u <= 1:
                interception_point = calculate_intersection_point_vec(p, t, r)
                distances = calculate_distance_to_intersection(interception_point, p)
                inter.append([True, interception_point, distances])

    return inter

# Conversão de Ângulos para Segmentos do LiDAR
@njit
def lidar_to_segment(robot_x: float, robot_y: float, lidar_range: float, angle: float) -> List[Tuple[float, float]]:
    return [
        (robot_x, robot_y),
        (robot_x + lidar_range * np.cos(angle), robot_y + lidar_range * np.sin(angle)),
    ]

# Algoritmo de Ray Casting para Detecção de Pontos em Polígonos
@njit
def ray_casting(edges: list, xp: float, yp: float) -> bool:
    cnt = 0
    for edge in edges:
        (x1, y1, x2, y2) = edge
        if (yp < y1) != (yp < y2):
            xinters = (yp - y1) * (x2 - x1) / (y2 - y1 + 1e-12) + x1  # Evita divisão por zero
            if xp < xinters:
                cnt += 1
    return cnt % 2 == 1

# Verificação se um Ponto Está em um Polígono
def is_point_in_polygon(exterior: list, interiors: list, xp: float, yp: float) -> bool:
    if not ray_casting(exterior, xp, yp):
        return False
    for hole in interiors:
        if ray_casting(hole, xp, yp):
            return False
    return True

# Verificação se um Círculo Está Dentro de um Polígono
def is_circle_in_polygon(
    exterior: list,
    interiors: list,
    xc: float,
    yc: float,
    radius: float = 0.3,
    num_segments: int = 100,
) -> bool:
    angles = np.linspace(0, 2 * np.pi, num_segments)
    for angle in angles:
        xp = xc + radius * np.cos(angle)
        yp = yc + radius * np.sin(angle)
        if not is_point_in_polygon(exterior, interiors, xp, yp):
            return False
    return True

# Funções para Processar Interseções e Medições do LiDAR
def lidar_intersections(
    robot_x: float,
    robot_y: float,
    robot_theta: float,
    lidar_range: float,
    lidar_angles: np.ndarray,
    segments: List[Tuple[float, float, float, float]],
) -> List[Tuple[float, float]]:
    intersections = []
    for angle in lidar_angles:
        adjusted_angle = angle + robot_theta
        lidar_segment = lidar_to_segment(robot_x, robot_y, lidar_range, adjusted_angle)
        intersected = intersect_segment_with_polygon(lidar_segment, segments)
        if intersected:
            intercept, position = position_intersection(intersected)
            if intercept:
                intersections.append(position)
            else:
                intersections.append((0.0, 0.0))
        else:
            intersections.append((0.0, 0.0))
    return intersections

def lidar_intersections_map_generated(
    robot_x: float,
    robot_y: float,
    robot_theta: float,
    lidar_range: float,
    lidar_angles: np.ndarray,
    segments: List,
):

    intersections = []
    for i, angle in enumerate(lidar_angles):
        adjusted_angle = angle + robot_theta

        lidar_segment = lidar_to_segment(robot_x, robot_y, lidar_range, adjusted_angle)

        lidar_segment_transformed = [np.array(segmento) for segmento in lidar_segment]

        intersected = intersect_segment_with_polygon_map_generated(
            lidar_segment_transformed, segments
        )

        if intersected:
            intercept, position = position_intersection(intersected)

            if intercept:
                intersections.append(position)

            else:
                intersections.append((0.0, 0.0))

        else:
            intersections.append((0.0, 0.0))
            continue

    return intersections

def lidar_measurements(
    robot_x: float,
    robot_y: float,
    robot_theta: float,
    lidar_range: float,
    lidar_angles: np.ndarray,
    segments: List[Tuple[float, float, float, float]],
) -> List[float]:
    measurements = []
    for angle in lidar_angles:
        adjusted_angle = angle + robot_theta
        lidar_segment = lidar_to_segment(robot_x, robot_y, lidar_range, adjusted_angle)
        intersected = intersect_segment_with_polygon(lidar_segment, segments)
        if intersected:
            intercept, lrange = measurements_intersection(intersected)
            if intercept:
                measurements.append(lrange)
            else:
                measurements.append(lidar_range)
        else:
            measurements.append(lidar_range)
    return measurements


def lidar_measurements_map_generated(
    robot_x: float,
    robot_y: float,
    robot_theta: float,
    lidar_range: float,
    lidar_angles: np.ndarray,
    segments: List,
):
    measurements = []
    for i, angle in enumerate(lidar_angles):
        # Adjust the angle by adding the robot's orientation
        adjusted_angle = angle + robot_theta

        # Compute the LiDAR segment using the adjusted angle
        lidar_segment = lidar_to_segment(robot_x, robot_y, lidar_range, adjusted_angle)
        lidar_segment_transformed = [np.array(segment) for segment in lidar_segment]

        intersected = intersect_segment_with_polygon_map_generated(
            lidar_segment_transformed, segments
        )

        if intersected:
            intercept, ranges = measurements_intersection(intersected)

            if intercept:
                measurements.append(ranges)
            else:
                measurements.append(0.2)
        else:
            measurements.append(0.2)
            continue

    return measurements

def position_intersection(
    intersection: List[Tuple[bool, np.ndarray, float]]
) -> Tuple[bool, Tuple[float, float]]:
    if not intersection:
        return False, (0.0, 0.0)
    min_lrange_index = np.argmin([point[2] for point in intersection])
    inter_point = intersection[min_lrange_index][1]
    inter_point_rounded = (round(inter_point[0], 3), round(inter_point[1], 3))
    return True, inter_point_rounded

def measurements_intersection(intersection: List[Tuple[bool, np.ndarray, float]]) -> Tuple[bool, float]:
    if not intersection:
        return False, 0.0
    min_lrange_index = np.argmin([point[2] for point in intersection])
    lrange = round(intersection[min_lrange_index][2], 3)
    return True, lrange
