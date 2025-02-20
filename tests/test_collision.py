import numpy as np
from numba import types
from numba.typed import List

from rnl.engine.collisions import (
    _point_in_polygon,
    _point_in_ring,
    connect_polygon_points,
    convert_to_line_segments,
    convert_to_segments,
    extract_segment_from_polygon,
    filter_list_segment,
    is_counter_clockwise,
)


def test_filter_list_segment():
    segs = [(0.0, 0.0, 1.0, 1.0), (100.0, 100.0, 101.0, 101.0)]
    result = filter_list_segment(segs, 0.0, 0.0, 5.0)
    assert len(result) == 1
    assert result[0] == (0.0, 0.0, 1.0, 1.0)


def test_is_counter_clockwise():
    # De acordo com o comportamento atual:
    # [(0,0), (1,0), (1,1), (0,1)] retorna True,
    # [(0,0), (0,1), (1,1), (1,0)] retorna False.
    polygon_true = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    polygon_false = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
    assert is_counter_clockwise(polygon_true) is True
    assert is_counter_clockwise(polygon_false) is False


def test_point_in_ring():
    ring = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
    assert _point_in_ring(0.5, 0.5, ring) is True
    assert _point_in_ring(1.5, 0.5, ring) is False


def test_point_in_polygon():
    exterior = [(0.0, 0.0), (0.0, 4.0), (4.0, 4.0), (4.0, 0.0)]
    hole = [(1.0, 1.0), (1.0, 3.0), (3.0, 3.0), (3.0, 1.0)]
    # Cria um typed list para os buracos com o tipo: List[List[UniTuple(float64, 2)]]
    holes_empty = List.empty_list(types.ListType(types.UniTuple(types.float64, 2)))
    # Teste sem buracos:
    assert _point_in_polygon(0.5, 0.5, exterior, holes_empty) is True

    # Agora, cria um typed list para um buraco:
    hole_typed = List.empty_list(types.UniTuple(types.float64, 2))
    for pt in hole:
        hole_typed.append(pt)
    holes_typed = List.empty_list(types.ListType(types.UniTuple(types.float64, 2)))
    holes_typed.append(hole_typed)
    # Ponto dentro do buraco deve retornar False:
    assert _point_in_polygon(2.0, 2.0, exterior, holes_typed) is False


def test_convert_to_segments():
    polygon = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    segments = convert_to_segments(polygon)
    # A função retorna segmentos como tuplas de 4 floats, não como tuplas aninhadas.
    expected = [
        (0.0, 0.0, 1.0, 0.0),
        (1.0, 0.0, 1.0, 1.0),
        (1.0, 1.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, 0.0),
    ]
    assert segments == expected


def test_connect_polygon_points():
    polygon = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    segments = connect_polygon_points(polygon)
    expected = [
        (np.array([0.0, 0.0]), np.array([1.0, 0.0])),
        (np.array([1.0, 0.0]), np.array([1.0, 1.0])),
        (np.array([1.0, 1.0]), np.array([0.0, 1.0])),
        (np.array([0.0, 1.0]), np.array([0.0, 0.0])),
    ]
    for seg, exp in zip(segments, expected):
        assert np.allclose(seg[0], exp[0])
        assert np.allclose(seg[1], exp[1])


def test_convert_to_line_segments():
    # Para satisfazer a assertiva (mínimo 3 segmentos), usamos um triângulo.
    total_segments = [
        ((np.array([0.0, 0.0]), np.array([1.0, 0.0]))),
        ((np.array([1.0, 0.0]), np.array([0.5, 1.0]))),
        ((np.array([0.5, 1.0]), np.array([0.0, 0.0]))),
    ]
    line_segments = convert_to_line_segments(total_segments)
    expected = [
        (0.0, 0.0, 1.0, 0.0),
        (1.0, 0.0, 0.5, 1.0),
        (0.5, 1.0, 0.0, 0.0),
    ]
    assert line_segments == expected


def test_extract_segment_from_polygon():
    polygon = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    stack = [polygon]
    segments = extract_segment_from_polygon(stack)
    expected = [
        (0.0, 0.0, 1.0, 0.0),
        (1.0, 0.0, 1.0, 1.0),
        (1.0, 1.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, 0.0),
    ]
    assert segments == expected
