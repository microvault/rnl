#!/usr/bin/env python3
import math
from datetime import datetime
from pathlib import Path
import yaml
from rnl.environment.generate import Generator

CONFIG_PATH = Path("/Users/nicolasalan/microvault/rnl/data/map6/map6.yaml")
OUT_FILE    = Path("../rnl/ros/tb3_ws/src/playground/worlds/demo.world")

def generate_world(segments, out_file: Path) -> None:
    """Gera o arquivo .world a partir da lista de segmentos em metros (já escalados)."""
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
        if length < 1e-4:
            continue
        yaw = math.atan2(y2 - y1, x2 - x1)
        models.append(f"""
    <model name="wall_{i}">
      <static>true</static>
      <pose>{cx:.4f} {cy:.4f} 0 0 0 {yaw:.4f}</pose>
      <link name="link">
        <visual name="vis">
          <geometry><box><size>{length:.4f} 0.10 1.00</size></box></geometry>
        </visual>
        <collision name="col">
          <geometry><box><size>{length:.4f} 0.10 1.00</size></box></geometry>
        </collision>
      </link>
    </model>""")
    sdf = header + "\n".join(models) + "\n  </world>\n</sdf>\n"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(sdf)

def main() -> None:
    info = yaml.safe_load(open(CONFIG_PATH, "r"))
    res = float(info["resolution"])  # 0.05 m/pixel
    print(f"[DEBUG] resolução do mapa: {res} m/pixel")

    gen = Generator(mode="custom")
    _, segs, _ = gen.world(grid_length=0)

    SCALE_FACTOR = 1.0

    segs_scaled = [
        (x1 * SCALE_FACTOR, y1 * SCALE_FACTOR, x2 * SCALE_FACTOR, y2 * SCALE_FACTOR)
        for x1, y1, x2, y2 in segs
    ]

    xs_s = [v for seg in segs_scaled for v in (seg[0], seg[2])]
    ys_s = [v for seg in segs_scaled for v in (seg[1], seg[3])]
    cx_s, cy_s = (min(xs_s) + max(xs_s)) / 2, (min(ys_s) + max(ys_s)) / 2
    segs_centered = [
        (x1 - cx_s, y1 - cy_s, x2 - cx_s, y2 - cy_s)
        for x1, y1, x2, y2 in segs_scaled
    ]

    generate_world(segs_centered, OUT_FILE)

if __name__ == "__main__":
    main()
