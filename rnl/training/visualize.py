#!/usr/bin/env python3
import numpy as np
import time
import rerun as rr
import argparse
from rnl.training.utils import create_single_env

import os

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
            rr.LineStrips3D(
                seg_3d_wrapped,
                radii=0.05,
                colors=[(255, 192, 203)]
            )
        )

def main():
    parser = argparse.ArgumentParser("Multi-Environment Training (3D)")
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, "Multi-Environment Training - 3D")

    num_envs = 100
    rows = 6
    cols = 10

    envs = [create_single_env(i) for i in range(num_envs)]

    # Posiciona e desenha cada ambiente
    for i, env in enumerate(envs):
        row = i // cols
        col = i % cols
        offset_x = col * 5.0
        offset_y = row * 5.0
        env.reset()
        log_map_segments(env, i, offset_x=offset_x, offset_y=offset_y)

    total_steps = 500
    for step in range(total_steps):
        rr.set_time_sequence("frame", step)
        for i, env in enumerate(envs):
            row = i // cols
            col = i % cols
            offset_x = col * 5.0
            offset_y = row * 5.0

            action = np.random.randint(0, 3)
            state, reward, done, truncated, info = env.step(action)

            pos = (
                env.body.position.x + offset_x,
                env.body.position.y + offset_y,
            )
            target = (
                env.target_x + offset_x,
                env.target_y + offset_y,
            )

            # Robô
            rr.log(
                f"env_{i}/robot",
                rr.Points3D(
                    np.array([[pos[0], pos[1], 0.0]], dtype=np.float32),
                    radii=[0.105],
                    colors=[(0, 0, 255)]
                )
            )
            # Alvo
            rr.log(
                f"env_{i}/target",
                rr.Points3D(
                    np.array([[target[0], target[1], 0.0]], dtype=np.float32),
                    radii=[0.05],
                    colors=[(0, 255, 0)]
                )
            )

            if done or truncated:
                env.reset()
                log_map_segments(env, i, offset_x=offset_x, offset_y=offset_y)

        time.sleep(0.01)

    rr.script_teardown(args)

if __name__ == "__main__":
    main()
