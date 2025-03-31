#!/usr/bin/env python3
import argparse
import os
import time

import numpy as np
import rerun as rr

from rnl.training.utils import create_single_env

os.environ["KMP_WARNINGS"] = "0"


def log_map_segments(env, env_index, offset_x=0.0, offset_y=0.0):
    segments = env.segments
    for idx, seg in enumerate(segments):
        x0, y0, x1, y1 = seg
        seg_2d = np.array([[x0, y0], [x1, y1]])
        seg_2d[:, 0] += offset_x
        seg_2d[:, 1] += offset_y

        seg_3d = np.hstack((seg_2d, np.zeros((seg_2d.shape[0], 1))))
        seg_3d_wrapped = seg_3d[np.newaxis, ...]

        rr.log(
            f"env_{env_index}/map_segment_{idx}",
            rr.LineStrips3D(seg_3d_wrapped, radii=0.05, colors=[(255, 192, 203)]),
        )


def log_robot_and_target(env, env_index, offset_x=0.0, offset_y=0.0):
    # Posição do robô
    pos = (
        env.body.position.x + offset_x,
        env.body.position.y + offset_y,
    )
    # Posição do alvo
    target = (
        env.target_x + offset_x,
        env.target_y + offset_y,
    )

    rr.log(
        f"env_{env_index}/robot",
        rr.Points3D(
            np.array([[pos[0], pos[1], 0.0]], dtype=np.float32),
            radii=[0.105],
            colors=[(0, 0, 255)],
        ),
    )
    rr.log(
        f"env_{env_index}/target",
        rr.Points3D(
            np.array([[target[0], target[1], 0.0]], dtype=np.float32),
            radii=[0.05],
            colors=[(0, 255, 0)],
        ),
    )


def main():
    parser = argparse.ArgumentParser("Multi-Environment Training (3D)")
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, "Multi-Environment Training - 3D")

    num_envs = 10
    cols = 10

    # 1) Cria todos os ambientes
    envs = [create_single_env(i) for i in range(num_envs)]

    # 2) Reseta e desenha mapa de todos os envs (carregar tudo antes de iniciar)
    for i, env in enumerate(envs):
        env.reset()

        row = i // cols
        col = i % cols
        offset_x = col * 5.0
        offset_y = row * 5.0

        # Desenha o mapa estático (segmentos) do ambiente
        log_map_segments(env, i, offset_x=offset_x, offset_y=offset_y)

        # Desenha o robô e o alvo iniciais
        log_robot_and_target(env, i, offset_x=offset_x, offset_y=offset_y)

    # 4) Loop de steps
    total_steps = 1000
    for step in range(total_steps):
        rr.set_time_sequence("frame", step)

        for i, env in enumerate(envs):
            row = i // cols
            col = i % cols
            offset_x = col * 5.0
            offset_y = row * 5.0

            action = np.random.randint(0, 3)
            state, reward, done, truncated, info = env.step(action)

            # Atualiza posição do robô e do alvo
            log_robot_and_target(env, i, offset_x=offset_x, offset_y=offset_y)

            if done or truncated:
                env.reset()
                log_map_segments(env, i, offset_x=offset_x, offset_y=offset_y)
                log_robot_and_target(env, i, offset_x=offset_x, offset_y=offset_y)

        time.sleep(0.01)

    rr.script_teardown(args)


if __name__ == "__main__":
    main()
