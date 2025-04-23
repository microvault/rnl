import argparse
import math
from datetime import datetime

from rnl.environment.world import CreateWorld


def generate_gazebo_world_with_walls(segments):
    """
    segments: lista de tuplas (x1, y1, x2, y2) definindo cada parede.
    """
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")

    sdf_header = f"""<?xml version="1.0" ?>
<!-- Gerado em: {timestamp_str} -->
<sdf version="1.6">
  <world name="default">
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>
"""
    sdf_footer = """
  </world>
</sdf>
"""
    models = ""
    for i, seg in enumerate(segments):
        x1, y1, x2, y2 = seg
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx**2 + dy**2)
        angle = math.atan2(dy, dx)

        wall_model = f"""
    <model name="wall_{i}">
      <static>true</static>
      <pose>{cx} {cy} 0 0 0 {angle}</pose>
      <link name="link">
         <collision name="collision">
             <geometry>
                <box>
                  <size>{length} 0.1 1</size>
                </box>
             </geometry>
         </collision>
         <visual name="visual">
             <geometry>
                <box>
                  <size>{length} 0.1 1</size>
                </box>
             </geometry>
         </visual>
      </link>
    </model>
"""
        models += wall_model

    sdf_world = sdf_header + models + sdf_footer

    file_name = "../rnl/ros/src/playground/worlds/my_world.world"
    with open(file_name, "w") as f:
        f.write(sdf_world)
    print("File: ", file_name)


def main():
    parser = argparse.ArgumentParser(
        description="Gera mundo Gazebo com paredes a partir do mapa"
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--name_map",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    create_world = CreateWorld(
        folder=args.folder,
        name=args.name_map,
    )

    # Usa o método world para obter segmentos (modo exemplo)
    _, segments_from_world, _ = create_world.world(mode="easy-01")

    # Redimensiona e centraliza
    xs = [seg[0] for seg in segments_from_world] + [
        seg[2] for seg in segments_from_world
    ]
    ys = [seg[1] for seg in segments_from_world] + [
        seg[3] for seg in segments_from_world
    ]
    center_x, center_y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    scale = 0.2

    new_segments = []
    for x1, y1, x2, y2 in segments_from_world:
        nx1 = (x1 - center_x) * scale
        ny1 = (y1 - center_y) * scale
        nx2 = (x2 - center_x) * scale
        ny2 = (y2 - center_y) * scale
        new_segments.append((nx1, ny1, nx2, ny2))

    generate_gazebo_world_with_walls(new_segments)


if __name__ == "__main__":
    main()
