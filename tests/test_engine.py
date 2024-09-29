import pytest
import numpy as np
from rnl.engine.collision import is_circle_in_polygon


def test_is_circle_in_polygon():
    # Definir um polígono simples (quadrado)
    exterior = [(0, 0, 10, 0), (10, 0, 10, 10), (10, 10, 0, 10), (0, 10, 0, 0)]

    # Definir um buraco no polígono
    interiors = [[(4, 4, 6, 4), (6, 4, 6, 6), (6, 6, 4, 6), (4, 6, 4, 4)]]

    # Teste 1: Círculo completamente dentro do polígono
    assert is_circle_in_polygon(exterior, interiors, 2, 2, 1) == True

    # Teste 2: Círculo parcialmente fora do polígono
    assert is_circle_in_polygon(exterior, interiors, 0, 0, 1) == False

    # Teste 3: Círculo completamente fora do polígono
    assert is_circle_in_polygon(exterior, interiors, 15, 15, 1) == False

    # Teste 4: Círculo dentro do buraco
    assert is_circle_in_polygon(exterior, interiors, 5, 5, 0.5) == False

    # Teste 5: Círculo tocando o buraco
    assert is_circle_in_polygon(exterior, interiors, 5, 5, 1) == False

    # Teste 6: Círculo grande que engloba o buraco
    assert is_circle_in_polygon(exterior, interiors, 5, 5, 3) == False

    # Teste 7: Círculo tocando a borda do polígono
    assert is_circle_in_polygon(exterior, interiors, 10, 5, 1) == False

    # Teste 8: Círculo com raio zero (ponto)
    assert is_circle_in_polygon(exterior, interiors, 5, 5, 0) == True
