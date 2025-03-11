#!/usr/bin/env python3
import numpy as np
import time
import rerun as rr
import argparse
from rnl.environment.env import NaviEnv
from rnl.configs.actions import get_actions_class
from rnl.configs.rewards import RewardConfig
from rnl.configs.config import RobotConfig, SensorConfig, EnvConfig, RenderConfig

def create_env(i):
    robot_config = RobotConfig(
        base_radius=0.105,
        vel_linear=[-1.0, 1.0],
        vel_angular=[-0.5, 0.5],
        wheel_distance=0.5,
        weight=5.0,
        threshold=0.5,
        collision=0.3,
        path_model="None"
    )
    sensor_config = SensorConfig(
        fov=240.0,
        num_rays=36,
        min_range=0.1,
        max_range=5.0 + i
    )
    env_config = EnvConfig(
        scalar=30,
        grid_length=2.0,
        folder_map="",
        name_map="",
        timestep=1000
    )
    render_config = RenderConfig(
        controller=False,
        debug=True,
        plot=False
    )
    actions_cfg = get_actions_class("BalancedActions")()
    reward_cfg = RewardConfig(
        reward_type="all",
        params={
            "scale_orientation": 0.003,
            "scale_distance": 0.1,
            "scale_time": 0.01,
            "scale_obstacle": 0.001,
        },
        description="Reward baseado em todos os fatores"
    )
    env = NaviEnv(
        robot_config, sensor_config, env_config, render_config,
        use_render=False, actions_cfg=actions_cfg, reward_cfg=reward_cfg
    )
    return env

def log_map(env, env_index, offset_x=0.0):
    """
    Loga as paredes (segments) do mapa em 3D para o ambiente indicado.
    """
    segments = env.segments
    if not segments:
        print(f"[log_map] Env {env_index} sem segmentos para o mapa?!")
        return

    seg_list_3d = []
    for seg in segments:
        # Cada seg deve ser [(x1, y1), (x2, y2)]
        if hasattr(seg, '__iter__') and len(seg) == 2:
            x1, y1 = seg[0]
            x2, y2 = seg[1]
            seg_list_3d.append([
                [x1 + offset_x, y1, 0.0],
                [x2 + offset_x, y2, 0.0]
            ])

    if seg_list_3d:
        rr.log(
            f"env_{env_index}/map",
            rr.LineStrips3D(
                np.array(seg_list_3d, dtype=np.float32),
                radii=0.02,                 # Aumente se ficar muito fino
                colors=[(128, 128, 128)]    # Cor única para todas as paredes
            )
        )

def main():
    parser = argparse.ArgumentParser("Multi-Environment Training (3D)")
    rr.script_add_args(parser)
    args = parser.parse_args()

    rr.script_setup(args, "Multi-Environment Training - 3D")

    num_envs = 4
    envs = [create_env(i) for i in range(num_envs)]

    # ----------------------------------------------------------------
    # Reset inicial de todos os ambientes e log do mapa
    # ----------------------------------------------------------------
    for i, env in enumerate(envs):
        env.reset()
        # Se quiser separar ambientes (offset), defina algo como offset_x = i*5
        log_map(env, i, offset_x=i * 5.0)

    # ----------------------------------------------------------------
    # Loop principal de steps
    # ----------------------------------------------------------------
    total_steps = 1000
    for step in range(total_steps):
        # Ajusta o tempo/frame no Rerun
        rr.set_time_sequence("frame", step)

        for i, env in enumerate(envs):
            action = np.random.randint(0, 3)
            state, reward, done, truncated, info = env.step(action)

            pos = (env.body.position.x, env.body.position.y)
            target = (env.target_x, env.target_y)
            offset_x = i * 5.0  # separar ambientes visualmente

            # Robô
            rr.log(
                f"env_{i}/robot",
                rr.Points3D(
                    np.array([[pos[0] + offset_x, pos[1], 0.0]], dtype=np.float32),
                    radii=[0.105],
                    colors=[(0, 0, 255)]
                )
            )
            # Alvo (target)
            rr.log(
                f"env_{i}/target",
                rr.Points3D(
                    np.array([[target[0] + offset_x, target[1], 0.0]], dtype=np.float32),
                    radii=[0.05],
                    colors=[(0, 255, 0)]
                )
            )

            # Se o ambiente terminar, reset e log do mapa de novo (caso mude)
            if done or truncated:
                env.reset()
                log_map(env, i, offset_x=offset_x)

        time.sleep(0.01)

    rr.script_teardown(args)

if __name__ == "__main__":
    main()
