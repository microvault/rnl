import json

import numpy as np
from tqdm import trange
import time
import wandb
from rnl.configs.actions import get_actions_class
from rnl.configs.rewards import RewardConfig
from rnl.configs.strategys import get_strategy_dict
from rnl.engine.utils import clean_info, statistics
from rnl.engine.vector import make_vect_envs
from agilerl.utils.utils import (
    init_wandb,
    save_population_checkpoint,
    tournament_selection_and_mutation,
)

def training_loop(
    use_agents,
    num_envs,
    max_steps,
    evo_steps,
    eval_steps,
    eval_loop,
    env,
    pop,
    tournament,
    mutation,
    evaluator,
    robot_config,
    sensor_config,
    env_config,
    render_config,
    actions_type,
    reward_instance,
    wb,
    save_path,
    elite_path,
    checkpoint,
    overwrite_checkpoints,
    INIT_HP,
    MUT_P,
    wandb_api_key,
    save_elite
):
    total_steps = 0
    infos_list = []
    pbar = trange(max_steps, unit="step")
    justificativas_history = []
    checkpoint_count = 0
    pop_loss = [[] for _ in pop]
    new_reward_type = None
    new_params = None
    new_action_type = None

    if wb:
        init_wandb(
            algo="PPO",
            env_name="NaviEnv",
            init_hyperparams=INIT_HP,
            mutation_hyperparams=MUT_P,
            wandb_api_key=wandb_api_key,
            accelerator=None,
        )


    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        pop_episode_scores = []

        for agent in pop:
            state, info = env.reset()
            scores = np.zeros(num_envs)
            completed_episode_scores = []
            steps = 0
            pop_fps = []
            pop_episode_scores = []
            start_time = time.time()

            # Coleta de experiências
            for _ in range(-(evo_steps // -agent.learn_step)):
                states, actions, log_probs, rewards, dones, values = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                learn_steps = 0

                for _ in range(-(agent.learn_step // -num_envs)):
                    action, log_prob, _, value = agent.get_action(state)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    info = clean_info(info)
                    if info:
                        infos_list.append(info)

                    total_steps += num_envs
                    steps += num_envs
                    learn_steps += num_envs

                    states.append(state)
                    actions.append(action)
                    log_probs.append(log_prob)
                    rewards.append(reward)
                    dones.append(terminated)
                    values.append(value)

                    state = next_state
                    scores += np.array(reward)

                    for idx, (d, t) in enumerate(zip(terminated, truncated)):
                        if d or t:
                            completed_episode_scores.append(scores[idx])
                            agent.scores.append(scores[idx])
                            scores[idx] = 0

                pbar.update(learn_steps // len(pop))
                experiences = (
                    states,
                    actions,
                    log_probs,
                    rewards,
                    dones,
                    values,
                    next_state,
                )
                agent.learn(experiences)

            agent.steps[-1] += steps
            fps = steps / (time.time() - start_time)
            pop_fps.append(fps)
            pop_episode_scores.append(completed_episode_scores)

        # Exibe os resultados parciais
        print(f"--- Global steps {total_steps} ---")
        print(f"Steps: {[agent.steps[-1] for agent in pop]}")
        mean_scores = [
            np.mean(ep) if len(ep) > 0 else "0 completed episodes"
            for ep in pop_episode_scores
        ]
        print(f"Scores: {mean_scores}")
        fitnesses = [
            agent.test(env, swap_channels=False, max_steps=eval_steps, loop=eval_loop)
            for agent in pop
        ]
        print(f"Fitnesses: {['%.2f' % f for f in fitnesses]}")
        print(f"5 fitness avgs: {['%.2f' % np.mean(agent.fitness[-5:]) for agent in pop]}")

        if wb:
            wandb_dict = {
                "global_step": (total_steps),
                "fps": np.mean(pop_fps),
                "train/mean_score": np.mean(
                    [
                        mean_score
                        for mean_score in mean_scores
                        if not isinstance(mean_score, str)
                    ]
                ),
                "eval/mean_fitness": np.mean(fitnesses),
                "eval/best_fitness": np.max(fitnesses),
            }

            agent_loss_dict = {
                f"train/agent_{index}_loss": np.mean(loss_[-10:])
                for index, loss_ in enumerate(pop_loss)
            }
            wandb_dict.update(agent_loss_dict)

            wandb.log(wandb_dict)

        # Calcula estatísticas para a avaliação
        if use_agents:
            stats = {}
            for campo in [
                "obstacle_score",
                "orientation_score",
                "progress_score",
                "time_score",
                "dist",
                "alpha",
                "min_lidar",
                "max_lidar",
            ]:
                if any(campo in info for info in infos_list):
                    media, _, _, desvio = statistics(infos_list, campo)
                    stats[campo + "_mean"] = media
                    stats[campo + "_std"] = desvio

            # Chama avaliação durante o treinamento, passando o histórico de justificativas
            evaluation_result = evaluator.evaluate_training(
                get_strategy_dict(), stats, justificativas_history
            )
            # print("Resultado da avaliação LLM:", evaluation_result)
            try:
                new_config = evaluation_result if isinstance(evaluation_result, dict) else json.loads(evaluation_result)
            except Exception as e:
                print("Erro ao parsear JSON da avaliação LLM:", e)
                new_config = {}

            if isinstance(evaluation_result, dict) and "justify" in evaluation_result:
                history_entry = {
                    "stats": stats,
                    "justify": evaluation_result["justify"],
                    "config": new_config
                }
                justificativas_history.append(history_entry)
                # Mantém apenas os 5 últimos itens
                if len(justificativas_history) > 5:
                    justificativas_history.pop(0)

            infos_list = []

            try:
                new_config = (
                    evaluation_result
                    if isinstance(evaluation_result, dict)
                    else json.loads(evaluation_result)
                )
            except Exception as e:
                print("Erro ao parsear JSON da avaliação LLM:", e)
                new_config = {}

            # Atualiza o ambiente se houver nova estratégia
            if "strategy" in new_config:
                strategy = new_config["strategy"]

                # Atualiza action
                action_info = strategy.get("action", {})
                new_action_type = action_info.get("action_type", None)
                if new_action_type:
                    actions_type = get_actions_class(new_action_type)()
                    # print(f"Novo action_type definido: {new_action_type}")
                else:
                    print("Nenhum novo action_type retornado, mantendo o atual")

                # Atualiza reward
                reward_info = strategy.get("reward", {})
                if reward_info:
                    new_reward_type = reward_info.get(
                        "reward_type", reward_instance.reward_type
                    )
                    parameters_list = reward_info.get("parameters", [])
                    new_params = {
                        param.get("key"): param.get("value") for param in parameters_list
                    }
                    reward_instance = RewardConfig(
                        reward_type=new_reward_type,
                        params=new_params,
                        description=f"Reward configurado para {new_reward_type}",
                    )
                    # print(f"Novo reward configurado: {new_reward_type}, parâmetros: {new_params}")
                else:
                    print("Nenhuma nova configuração de reward retornada, mantendo o atual")

                # Recria o ambiente com os novos parâmetros
                env = make_vect_envs(
                    num_envs=num_envs,
                    robot_config=robot_config,
                    sensor_config=sensor_config,
                    env_config=env_config,
                    render_config=render_config,
                    use_render=False,
                    actions_type=actions_type,
                    reward_type=reward_instance,
                )
            else:
                print("Nenhuma estratégia retornada. Ambiente permanece inalterado.")

        # Seleção e mutação
        pop = tournament_selection_and_mutation(
            population=pop,
            tournament=tournament,
            mutation=mutation,
            env_name="NavEnv",
            algo="PPO",
            elite_path=elite_path,
            save_elite=save_elite,
            accelerator=None,
        )
        for agent in pop:
            agent.steps.append(agent.steps[-1])

        if pop[0].steps[-1] // checkpoint > checkpoint_count:
            save_population_checkpoint(
                population=pop,
                save_path=save_path,
                overwrite_checkpoints=overwrite_checkpoints,
                accelerator=None,
            )
            checkpoint_count += 1

    print(f"Novo reward configurado: {new_reward_type}, parâmetros: {new_params}")
    print(f"Novo action_type definido: {new_action_type}")
    pbar.close()
    env.close()
    return infos_list, env
