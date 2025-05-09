import numpy as np


def statistics(info_list, field):
    values = [info[field] for info in info_list if field in info]
    if not values:
        return None, None, None, None
    mean_value = np.mean(values)
    min_value = np.min(values)
    max_value = np.max(values)
    std_deviation = np.std(values)
    return mean_value, min_value, max_value, std_deviation


def evaluate_agent(
    agent,
    env,
    num_episodes: int = 10,
    max_steps_ep: int = 500,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
]:
    """
    Avalia o agente e devolve tudo já em **percentual (0-100)**.

    Retorna:
        (success_pct,
         unsafe_pct,
         angular_pct,
         ep_len_pct,
         collision_steps_pct,
         goal_steps_pct)
    """
    num_goals          = 0
    unsafe_steps       = 0
    angular_steps      = 0
    total_timesteps    = 0

    total_steps_col    = 0
    count_collision    = 0
    total_steps_goal   = 0
    count_goal         = 0
    info = {}

    for ep in range(num_episodes):
        state, _ = env.reset()
        done = truncated = False

        # executa episódio
        while not done and not truncated:
            action, _ = agent.predict(state, deterministic=False)
            state, _, done, truncated, info = env.step(action)

        # usa os dados do último passo
        steps_collision  = info.get("steps_to_collision", 0)
        steps_goal       = info.get("steps_to_goal", 0)
        unsafe_steps    += info.get("steps_unsafe_area", 0)
        angular_steps   += info.get("steps_command_angular", 0)
        total_timesteps += info.get("total_timestep", 0)

        if steps_collision > 0:
            total_steps_col += steps_collision
            count_collision += 1
        if steps_goal > 0:
            total_steps_goal += steps_goal
            count_goal      += 1
            if steps_collision == 0:
                num_goals += 1

    # ---------- métricas em % ----------
    success_pct  = (num_goals / num_episodes) * 100
    denom_steps  = num_episodes * max_steps_ep
    unsafe_pct   = (unsafe_steps  / denom_steps) * 100
    angular_pct  = (angular_steps / denom_steps) * 100
    ep_len_pct   = (total_timesteps / denom_steps) * 100
    collision_steps_pct = (
        (total_steps_col / denom_steps) * 100 if count_collision else 0
    )
    goal_steps_pct = (
        (total_steps_goal / denom_steps) * 100 if count_goal else 0
    )

    return (
        round(success_pct, 2),
        round(unsafe_pct, 2),
        round(angular_pct, 2),
        round(ep_len_pct, 2),
        round(collision_steps_pct, 2),
        round(goal_steps_pct, 2),
    )
