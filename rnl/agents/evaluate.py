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
    agent, env, num_episodes=10
) -> tuple[float, int, float, float, float, float, float]:
    num_goals = 0
    unsafe = 0
    command_angular = 0
    total_timesteps = 0

    total_steps_collision = 0
    count_collision = 0
    total_steps_goal = 0
    count_goal = 0

    for ep in range(num_episodes):
        state, info = env.reset()
        done = False
        truncated = False

        while not done and not truncated:
            action, _ = agent.predict(state, deterministic=False)
            state, reward, done, truncated, _ = env.step(action)

        _, final_info = env.reset()

        steps_collision = final_info.get("steps_to_collision", 0)
        steps_goal = final_info.get("steps_to_goal", 0)
        steps_unsafe_area = final_info.get("steps_unsafe_area", 0)
        steps_command_angular = final_info.get("steps_command_angular", 0)
        total_timestep = final_info.get("total_timestep", 0)

        unsafe += steps_unsafe_area
        command_angular += steps_command_angular
        total_timesteps += total_timestep

        if steps_collision > 0:
            total_steps_collision += steps_collision
            count_collision += 1

        if steps_goal > 0:
            total_steps_goal += steps_goal
            count_goal += 1

        if steps_collision == 0 and steps_goal > 0:
            num_goals += 1

    success_percentage = (num_goals / num_episodes) * 100 if num_episodes else 0
    percentage_angular = (
        command_angular / total_timesteps if total_timesteps != 0 else 0
    )
    percentage_unsafe = unsafe / total_timesteps if total_timesteps != 0 else 0
    ep_mean_length = total_timesteps / num_episodes if num_episodes else 0
    avg_collision_steps = (
        total_steps_collision / count_collision if count_collision else 0
    )
    avg_goal_steps = total_steps_goal / count_goal if count_goal else 0

    return (
        success_percentage,
        total_timesteps,
        round(percentage_unsafe, 2),
        round(percentage_angular, 2),
        ep_mean_length,
        avg_collision_steps,
        avg_goal_steps,
    )
