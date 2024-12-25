import warnings
from datetime import datetime
from typing import List

import numpy as np
import wandb
from gymnasium.vector import AsyncVectorEnv
from tqdm import trange

from rnl.algorithms.rainbow import RainbowDQN
from rnl.components.replay_buffer import MultiStepReplayBuffer, PrioritizedReplayBuffer
from rnl.components.sampler import Sampler
from rnl.hpo.mutation import Mutations
from rnl.hpo.tournament import TournamentSelection


def train_off_policy(
    config: dict,
    env: AsyncVectorEnv,
    pop: List[RainbowDQN],
    memory: PrioritizedReplayBuffer,
    max_steps: int,
    evo_steps: int,
    eval_steps,
    eval_loop: int,
    learning_delay,
    eps_start: float,
    eps_end: float,
    eps_decay: float,
    target: int,
    n_step_memory: MultiStepReplayBuffer,
    tournament: TournamentSelection,
    mutation: Mutations,
    checkpoint: int,
    checkpoint_path: str,
    overwrite_checkpoints: bool,
    wb: bool,
    wandb_api_key: str,
):

    if wb:
        if not hasattr(wandb, "api"):
            if wandb_api_key is not None:
                wandb.login(key=wandb_api_key)
            else:
                warnings.warn("Must login to wandb with API key.")

        wandb.init(
            project="rnl",
            name="train-{}".format(datetime.now().strftime("%m%d%Y%H%M%S")),
            config=config,
        )

    num_envs = env.num_envs

    save_path = (
        checkpoint_path.split(".pt")[0]
        if checkpoint_path is not None
        else "rnl-{}".format(datetime.now().strftime("%m%d%Y%H%M%S"))
    )

    sampler = Sampler(per=True, n_step=False, memory=memory)
    n_step_sampler = Sampler(per=False, n_step=True, memory=n_step_memory)

    print("\nTraining...")

    bar_format = "{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]"
    pbar = trange(max_steps, unit="step", bar_format=bar_format, ascii=True)

    pop_loss = [[] for _ in pop]
    pop_fitnesses = []
    total_steps = 0
    loss = None
    checkpoint_count = 0

    # RL training loop
    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        pop_episode_scores = []
        for agent_idx, agent in enumerate(pop):
            state = env.reset()[0]
            scores = np.zeros(num_envs)
            completed_episode_scores, losses = [], []
            steps = 0

            train_actions_hist = [0] * agent.action_dim

            for idx_step in range(evo_steps // num_envs):

                action = agent.get_action(state)

                for a in action:
                    if not isinstance(a, int):
                        a = int(a)
                    train_actions_hist[a] += 1

                # Act in environment
                next_state, reward, done, trunc, _ = env.step(action)
                scores += np.array(reward)

                reset_noise_indices = []
                for idx, (d, t) in enumerate(zip(done, trunc)):
                    if d or t:
                        completed_episode_scores.append(scores[idx])
                        agent.scores.append(scores[idx])
                        scores[idx] = 0
                        reset_noise_indices.append(idx)

                total_steps += num_envs
                steps += num_envs

                # Save experience to replay buffer
                one_step_transition = n_step_memory.save_to_step_memory_vect_envs(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )

                memory.save_to_memory_vect_envs(*one_step_transition)

                fraction = min(
                    ((agent.steps[-1] + idx_step + 1) * num_envs / max_steps), 1.0
                )
                agent.beta += fraction * (1.0 - agent.beta)

                # Learn according to learning frequency
                if agent.learn_step > num_envs:
                    learn_step = agent.learn_step // num_envs
                    if (
                        idx_step % learn_step == 0
                        and len(memory) >= agent.batch_size
                        and memory.counter > learning_delay
                    ):
                        if sampler.per:
                            experiences = sampler._sample_per(
                                agent.batch_size, beta=agent.beta
                            )
                        else:
                            experiences = sampler._sample_standard(agent.batch_size)
                        if n_step_memory is not None:
                            n_step_experiences = n_step_sampler._sample_n_step(
                                idxs=experiences[6]
                            )
                            experiences += n_step_experiences
                        loss, idxs, priorities = agent.learn_dqn(experiences)
                        memory.update_priorities(idxs, priorities)

                elif (
                    len(memory) >= agent.batch_size and memory.counter > learning_delay
                ):
                    for _ in range(num_envs // agent.learn_step):
                        # Sample replay buffer
                        # Learn according to agent's RL algorithm
                        if sampler.per:
                            experiences = sampler._sample_per(
                                agent.batch_size, beta=agent.beta
                            )
                        else:
                            experiences = sampler._sample_standard(agent.batch_size)

                        if n_step_memory is not None:
                            n_step_experiences = n_step_sampler._sample_n_step(
                                idxs=experiences[6]
                            )
                            experiences += n_step_experiences
                        loss, idxs, priorities = agent.learn_dqn(experiences)
                        memory.update_priorities(idxs, priorities)

                if loss is not None:
                    losses.append(loss)

                state = next_state

            agent.steps[-1] += steps
            pbar.update(evo_steps // len(pop))

            pop_episode_scores.append(completed_episode_scores)

            if len(losses) > 0:
                mean_loss = np.mean(losses)
                pop_loss[agent_idx].append(mean_loss)

        # Evaluate population
        fitnesses = [
            agent.test(env, max_steps=eval_steps, loop=eval_loop) for agent in pop
        ]
        pop_fitnesses.append(fitnesses)
        mean_scores = [
            (
                float(np.mean(episode_scores))
                if len(episode_scores) > 0
                else "0 completed episodes"
            )
            for episode_scores in pop_episode_scores
        ]

        if wb:
            wandb_dict = {
                "global_step": total_steps,
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

            # Create the loss dictionaries
            # if algo in ["RainbowDQN", "DQN"]:
            actor_loss_dict = {
                f"train/agent_{index}_actor_loss": np.mean(loss[-10:])
                for index, loss in enumerate(pop_loss)
            }
            wandb_dict.update(actor_loss_dict)

            # if algo in ["DQN", "Rainbow DQN"]:
            #     train_actions_hist = [
            #         freq / sum(train_actions_hist) for freq in train_actions_hist
            #     ]
            #     train_actions_dict = {
            #         f"train/action_{index}": action
            #         for index, action in enumerate(train_actions_hist)
            #     }
            #     wandb_dict.update(train_actions_dict)

            wandb.log(wandb_dict)

        # Update step counter
        for agent in pop:
            agent.steps.append(agent.steps[-1])

        # Early stop if consistently reaches target
        if target is not None:
            if (
                np.all(
                    np.greater([np.mean(agent.fitness[-10:]) for agent in pop], target)
                )
                and len(pop[0].steps) >= 100
            ):
                if wb:
                    wandb.finish()
                return pop, pop_fitnesses

        # Tournament selection and population mutation
        elite, pop = tournament.select(pop)
        pop = mutation.mutation(pop)

        elite_save_path = "checkpoints/rnl-elite_rainbow"
        elite.save_checkpoint(f"{elite_save_path}.pt")

        fitness = ["%.2f" % fitness for fitness in fitnesses]
        avg_fitness = ["%.2f" % safe_mean(agent.fitness[-5:]) for agent in pop]
        avg_score = ["%.2f" % safe_mean(agent.scores[-10:]) for agent in pop]
        agents = [agent.index for agent in pop]
        num_steps = [agent.steps[-1] for agent in pop]
        pbar.update(0)

        print(
            f"""
            --- Global Steps {total_steps} ---
            Fitness:\t\t{fitness}
            Score:\t\t{mean_scores}
            5 fitness avgs:\t{avg_fitness}
            10 score avgs:\t{avg_score}
            Agents:\t\t{agents}
            Steps:\t\t{num_steps}
            """,
            end="\r",
        )

        # Save model checkpoint
        if pop[0].steps[-1] // checkpoint > checkpoint_count:
            for i, agent in enumerate(pop):
                current_checkpoint_path = (
                    f"{save_path}_{i}.pt"
                    if overwrite_checkpoints
                    else f"{save_path}_{i}_{agent.steps[-1]}.pt"
                )
                agent.save_checkpoint(current_checkpoint_path)
            print("Saved checkpoint.")
            checkpoint_count += 1

    if wb:
        wandb.finish()

    pbar.close()
    return pop, pop_fitnesses


def safe_mean(values, default=0.00):
    if len(values) == 0:
        return default
    return np.mean(values)
