#!/usr/bin/env python3
# build_world.py – gera um SDF 2,20 m × 2,15 m (grade de 5 cm)

import math
from datetime import datetime
from pathlib import Path
from rnl.environment.generate import Generator

# parâmetros fixos
RES = 0.05         # m/px
LEN_X, LEN_Y = 2.20, 2.15
CELLS_X = int(LEN_X / RES) + 1   # 45
CELLS_Y = int(LEN_Y / RES) + 1   # 44
OUT_FILE = Path("../rnl/ros/src/playground/worlds/demo.world")


def generate_world(segments, out_file: Path) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"""<?xml version="1.0" ?>
<!-- Gerado em {now} -->
<sdf version="1.6">
  <world name="default">
    <include><uri>model://ground_plane</uri></include>
    <include><uri>model://sun</uri></include>
"""
    models = []
    for i, (x1, y1, x2, y2) in enumerate(segments):
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 1e-4:  # ignora segmentos degenerados
            continue
        yaw = math.atan2(y2 - y1, x2 - x1)
        models.append(
            f"""
    <model name="wall_{i}">
      <static>true</static>
      <pose>{cx} {cy} 0 0 0 {yaw}</pose>
      <link name="link">
        <visual name="vis">
          <geometry><box><size>{length} 0.10 1.00</size></box></geometry>
        </visual>
        <collision name="col">
          <geometry><box><size>{length} 0.10 1.00</size></box></geometry>
        </collision>
      </link>
    </model>"""
        )

    sdf = header + "\n".join(models) + "\n  </world>\n</sdf>\n"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(sdf)
    print("⚡  Mundo salvo em:", out_file.resolve())


def main() -> None:
    gen = Generator(mode="custom")
    _, segs, _ = gen.world(
        grid_length=0,
        grid_length_x=CELLS_X,
        grid_length_y=CELLS_Y,
        resolution=RES,
    )

    # centraliza no (0,0)
    xs = [p for s in segs for p in (s[0], s[2])]
    ys = [p for s in segs for p in (s[1], s[3])]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    segs_centered = [(x1 - cx, y1 - cy, x2 - cx, y2 - cy) for x1, y1, x2, y2 in segs]

    generate_world(segs_centered, OUT_FILE)


if __name__ == "__main__":
    main()
