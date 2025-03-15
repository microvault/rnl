import numpy as np

def statistics(info_list, field):
    values = [info[field] for info in info_list if field in info]
    if not values:  # se a lista estiver vazia, retorne None ou valores padrão
        return None, None, None, None
    mean_value = np.mean(values)
    min_value = np.min(values)
    max_value = np.max(values)
    std_deviation = np.std(values)
    return mean_value, min_value, max_value, std_deviation

def evaluate_agent(agent, env, num_episodes=10):
    """
    Avalia o agente treinado (em vez de ações aleatórias) e mostra a porcentagem
    de vezes que o robô chegou ao objetivo sem colidir, ao longo de 'num_episodes' episódios.
    """
    num_goals = 0

    for ep in range(num_episodes):
        state, info = env.reset()
        done = False
        truncated = False

        while not done and not truncated:
            # Obtém a ação do agente treinado
            action, _ = agent.predict(state, deterministic=True)
            state, reward, done, truncated, _ = env.step(action)

        # Reseta para coletar os steps_to_collision e steps_to_goal finais
        _, final_info = env.reset()

        steps_collision = final_info.get("steps_to_collision", 0)
        steps_goal = final_info.get("steps_to_goal", 0)

        # Se houve colisão, steps_collision > 0; se não, consideramos chegada ao goal
        if steps_collision == 0 and steps_goal > 0:
            num_goals += 1

    success_percentage = (num_goals / num_episodes) * 100

    return success_percentage
