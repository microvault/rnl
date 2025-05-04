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
    max_steps_ep: int = 1000,        # ← passo máx. por episódio
) -> tuple[float, int, float, float, float, float, float]:
    """
    Retorna:
      (success_pct, total_steps, unsafe_pct, angular_pct,
       ep_mean_len, avg_collision_steps, avg_goal_steps)

    * unsafe_pct  e angular_pct são relativos ao total possível
      (num_episodes * max_steps_ep) em vez do total real executado.
    """
    num_goals          = 0
    unsafe_steps       = 0
    angular_steps      = 0
    total_timesteps    = 0

    total_steps_col    = 0
    count_collision    = 0
    total_steps_goal   = 0
    count_goal         = 0

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = truncated = False

        while not done and not truncated:
            action, _ = agent.predict(state, deterministic=False)
            state, _, done, truncated, _ = env.step(action)

        _, final_info         = env.reset()
        steps_collision       = final_info.get("steps_to_collision", 0)
        steps_goal            = final_info.get("steps_to_goal", 0)
        unsafe_steps         += final_info.get("steps_unsafe_area", 0)
        angular_steps        += final_info.get("steps_command_angular", 0)
        total_timesteps      += final_info.get("total_timestep", 0)

        if steps_collision > 0:
            total_steps_col  += steps_collision
            count_collision  += 1
        if steps_goal > 0:
            total_steps_goal += steps_goal
            count_goal       += 1
            if steps_collision == 0:
                num_goals += 1

    success_pct         = (num_goals / num_episodes) * 100
    max_steps_all       = num_episodes * max_steps_ep
    unsafe_pct          = (unsafe_steps  / max_steps_all) * 100
    angular_pct         = (angular_steps / max_steps_all) * 100
    ep_mean_len         = total_timesteps / num_episodes
    avg_collision_steps = total_steps_col  / count_collision if count_collision else 0
    avg_goal_steps      = total_steps_goal / count_goal      if count_goal      else 0

    return (
        round(success_pct, 2),
        total_timesteps,
        round(unsafe_pct, 2),
        round(angular_pct, 2),
        ep_mean_len,
        avg_collision_steps,
        avg_goal_steps,
    )
