#!/usr/bin/env python3
import numpy as np
import time
import rerun as rr
import argparse
import gymnasium as gym
from rnl.environment.env import NaviEnv
from rnl.configs.actions import get_actions_class
from rnl.configs.rewards import RewardConfig
from rnl.configs.config import RobotConfig, SensorConfig, EnvConfig, RenderConfig

def create_env(num_envs):
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

    def make_env(i):
        def _init():
            env = NaviEnv(
                robot_config,
                sensor_config,
                env_config,
                render_config,
                False,
                actions_cfg=actions_cfg,
                reward_cfg=reward_cfg,
            )
            env.reset(seed=13 + i)
            return env

        return _init

    return gym.vector.AsyncVectorEnv([make_env(i) for i in range(num_envs)])

def log_map_segments(env, env_index, offset_x=0.0):
    """
    Converte os segmentos 2D do mapa para 3D e plota cada um em rosa.
    Cada segmento é embrulhado num array extra para ter o shape (1, N, 3).
    """
    segments = env.segments
    if not segments:
        print(f"Env {env_index} sem segmentos!")
        return

    for idx, seg in enumerate(segments):
        seg_np = np.array(seg)
        if seg_np.ndim != 2 or seg_np.shape[1] != 2:
            continue

        # Aplica offset e acrescenta a coordenada z=0
        seg_2d = seg_np + np.array([offset_x, 0])
        seg_3d = np.hstack((seg_2d, np.zeros((seg_2d.shape[0], 1))))
        # Embrulha para ter shape (1, N, 3)
        seg_3d_wrapped = np.array([seg_3d])
        rr.log(
            f"env_{env_index}/map_segment_{idx}",
            rr.LineStrips3D(
                seg_3d_wrapped,
                radii=0.05,          # tente aumentar o radius se necessário
                colors=[(255, 192, 203)]  # rosa
            )
        )

def main():
    parser = argparse.ArgumentParser("Multi-Environment Training (3D)")
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, "Multi-Environment Training - 3D")

    num_envs = 1
    envs = [create_env(i) for i in range(num_envs)]

    # Reset e log do mapa para cada ambiente (aplicando offset para separar visualmente)
    for i, env in enumerate(envs):
        env.reset()
        offset = i * 5.0
        log_map_segments(env, i, offset_x=offset)

    total_steps = 50
    for step in range(total_steps):
        rr.set_time_sequence("frame", step)
        for i, env in enumerate(envs):
            action = np.random.randint(0, 3)
            state, reward, done, truncated, info = env.step(action)

            pos = (env.body.position.x, env.body.position.y)
            target = (env.target_x, env.target_y)
            offset = i * 5.0

            # Log do robô
            rr.log(
                f"env_{i}/robot",
                rr.Points3D(
                    np.array([[pos[0] + offset, pos[1], 0.0]], dtype=np.float32),
                    radii=[0.105],
                    colors=[(0, 0, 255)]
                )
            )
            # Log do alvo
            rr.log(
                f"env_{i}/target",
                rr.Points3D(
                    np.array([[target[0] + offset, target[1], 0.0]], dtype=np.float32),
                    radii=[0.05],
                    colors=[(0, 255, 0)]
                )
            )

            if done or truncated:
                env.reset()
                log_map_segments(env, i, offset_x=offset)

        time.sleep(0.01)

    rr.script_teardown(args)

if __name__ == "__main__":
    main()
