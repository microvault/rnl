import math
import argparse

from datetime import datetime
from rnl.environment.world import CreateWorld

def generate_gazebo_world_with_walls(segments):
    """
    segments: lista de tuplas (x1, y1, x2, y2) definindo cada parede.
    """
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    filename_timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

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

    file_name = f"my_gazebo_world_{filename_timestamp}.world"
    with open(file_name, "w") as f:
        f.write(sdf_world)
    print("Arquivo de mundo gerado:", file_name)


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
    _, segments_from_world, _ = create_world.world(mode="medium-07")
    # Se não houver segmentos, usa uma lista padrão
    segments = segments_from_world if segments_from_world else [
        (0, 0, 5, 0),
        (5, 0, 5, 5),
        (5, 5, 0, 5),
        (0, 5, 0, 0)
    ]

    generate_gazebo_world_with_walls(segments)


if __name__ == "__main__":
    main()
