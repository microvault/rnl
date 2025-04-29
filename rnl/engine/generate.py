#!/usr/bin/env python3
# generate_world.py – cria um world SDF a partir de um mapa Cartographer (map.yaml)

import argparse
import os
import math
from datetime import datetime
from pathlib import Path

import yaml
import numpy as np
from skimage.io import imread
from skimage.measure import find_contours
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely import affinity


def extract_segment_from_polygon(polys):
    """
    Função auxiliar que converte polígonos (lista de pontos) em segmentos de parede.
    Cada segmento é (x1,y1,x2,y2).
    """
    segments = []
    for coords in polys:
        # percorre cada aresta do polígono
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            segments.append((x1, y1, x2, y2))
    return segments


def load_map(map_yaml_path, threshold=0.65):
    """
    Lê o arquivo map.yaml do Cartographer e retorna os segmentos centrados.
    """
    # carrega info do YAML
    with open(map_yaml_path, 'r') as f:
        info = yaml.safe_load(f)

    res = float(info['resolution'])
    ox, oy, oyaw = info['origin']
    image_path = os.path.join(os.path.dirname(map_yaml_path), info['image'])

    # lê imagem (PGM/PNG)
    img = imread(image_path)
    if info.get('negate', 0):
        img = 255 - img
    occ = img < int(threshold * 255)
    h, w = occ.shape

    # extrai contornos
    contours = find_contours(occ.astype(float), 0.5)

    # converte para polígonos no espaço métrico
    polys = []
    for c in contours:
        pts = np.stack([
            ox + c[:, 1] * res,
            oy + (h - c[:, 0]) * res
        ], axis=1)
        polys.append(Polygon(pts).buffer(0))

    # união e ajuste de rotação
    poly = unary_union(polys)
    if oyaw:
        cx, cy = ox + (w * res) / 2, oy + (h * res) / 2
        poly = affinity.rotate(poly, math.degrees(oyaw), origin=(cx, cy), use_radians=False)

    # extrai segmentos
    stacks = []
    if isinstance(poly, MultiPolygon):
        for p in poly.geoms:
            stacks.append(np.asarray(p.exterior.coords, dtype=np.float64))
    else:
        stacks.append(np.asarray(poly.exterior.coords, dtype=np.float64))

    segments = extract_segment_from_polygon(stacks)

    # centraliza segmentos em (0,0)
    xs = [p for s in segments for p in (s[0], s[2])]
    ys = [p for s in segments for p in (s[1], s[3])]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    centered = [(x1 - cx, y1 - cy, x2 - cx, y2 - cy) for x1, y1, x2, y2 in segments]
    return centered


def generate_world(segments, out_file: Path):
    """
    Gera um arquivo SDF (.world) com muros baseados nos segmentos.
    """
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    header = f'''<?xml version="1.0" ?>
<!-- Gerado em {now} -->
<sdf version="1.6">
  <world name="default">
    <include><uri>model://ground_plane</uri></include>
    <include><uri>model://sun</uri></include>\n'''
    models = []

    for i, (x1, y1, x2, y2) in enumerate(segments):
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 0.05:
            continue  # filtra paredes < 5cm
        yaw = math.atan2(y2 - y1, x2 - x1)
        cz = 0.5  # metade da altura (1m)
        thickness = 0.10
        height = 1.00
        models.append(f'''
    <model name="wall_{i}">
      <static>true</static>
      <pose>{cx:.3f} {cy:.3f} {cz:.3f} 0 0 {yaw:.6f}</pose>
      <link name="link">
        <visual name="vis">
          <geometry><box><size>{length:.3f} {thickness:.3f} {height:.3f}</size></box></geometry>
        </visual>
        <collision name="col">
          <geometry><box><size>{length:.3f} {thickness:.3f} {height:.3f}</size></box></geometry>
        </collision>
      </link>
    </model>''')

    footer = '\n  </world>\n</sdf>\n'
    sdf = header + ''.join(models) + footer
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(sdf)
    print(f"⚡ Mundo salvo em: {out_file.resolve()}")


def main():
    map_yaml = "/Users/nicolasalan/microvault/rnl/data/map6/map6.yaml"
    threshold = 0.65
    OUT_FILE = "../rnl/ros/tb3_ws/src/playground/worlds/demo.world"

    segments = load_map(map_yaml, threshold=threshold)
    generate_world(segments, Path(OUT_FILE))

if __name__ == '__main__':
    main()
