import numpy as np
from numba import njit
import pytest

@njit(fastmath=True, cache=True)
def filter_list_segment(segs, x, y, max_range):
    segments_inside = []
    region_center = np.array([x, y])
    for x1, y1, x2, y2 in segs:
        seg_ends = np.array([[x1, y1], [x2, y2]])
        distances = np.sqrt(np.sum((seg_ends - region_center) ** 2, axis=1))
        if np.all(distances <= max_range):
            segments_inside.append((x1, y1, x2, y2))
    return segments_inside

def test_todos_segmentos_dentro():
    segs = [(1, 1, 2, 2), (1.5, 1.5, 2.5, 2.5)]
    x, y = 2, 2
    max_range = 1.5
    # Ambos os segmentos estão totalmente dentro do range
    resultado = filter_list_segment(segs, x, y, max_range)
    assert resultado == segs

def test_alguns_segmentos_fora():
    segs = [(1, 1, 2, 2), (5, 5, 6, 6)]
    x, y = 2, 2
    max_range = 1.5
    # Somente o primeiro segmento está dentro do range
    resultado = filter_list_segment(segs, x, y, max_range)
    assert resultado == [(1, 1, 2, 2)]
