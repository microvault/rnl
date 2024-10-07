from rnl.engine.collision import is_circle_in_polygon


def test_is_circle_in_polygon():
    exterior = [(0, 0, 10, 0), (10, 0, 10, 10), (10, 10, 0, 10), (0, 10, 0, 0)]
    interiors = [[(4, 4, 6, 4), (6, 4, 6, 6), (6, 6, 4, 6), (4, 6, 4, 4)]]
    assert is_circle_in_polygon(exterior, interiors, 2, 2, 1)
    assert not is_circle_in_polygon(exterior, interiors, 0, 0, 1)
    assert not is_circle_in_polygon(exterior, interiors, 15, 15, 1)
    assert not is_circle_in_polygon(exterior, interiors, 5, 5, 0.5)
    assert not is_circle_in_polygon(exterior, interiors, 5, 5, 1)
    assert not is_circle_in_polygon(exterior, interiors, 10, 5, 1)
