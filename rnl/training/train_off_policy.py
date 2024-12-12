import os
import pdb

# import warnings
from datetime import datetime
from typing import List

import numpy as np
from gymnasium.vector import AsyncVectorEnv
from torch.utils.data import DataLoader
from tqdm import trange

from rnl.algorithms.rainbow import RainbowDQN
from rnl.components.replay_buffer import MultiStepReplayBuffer, PrioritizedReplayBuffer

# import wandb
from rnl.components.replay_data import ReplayDataset
from rnl.components.sampler import Sampler
from rnl.hpo.mutation import Mutations
from rnl.hpo.tournament import TournamentSelection


def train_off_policy(
    env: AsyncVectorEnv,
    pop: List[RainbowDQN],
    memory: PrioritizedReplayBuffer,
    max_steps: int,  # 1000000
    evo_steps: int,  # 10000
    eval_steps,  # None
    eval_loop: int,  # 1
    learning_delay,  # 0
    eps_start: float,  # 1.0
    eps_end: float,  # 0.1
    eps_decay: float,  # 0.995
    target: int,  # 100
    n_step: bool,
    per: bool,
    n_step_memory: MultiStepReplayBuffer,
    tournament: TournamentSelection,
    mutation: Mutations,
    checkpoint: int = 10,
    checkpoint_path: str = None,
    overwrite_checkpoints: bool = False,
    save_elite: bool = False,
    elite_path=None,
    wb: bool = False,
    verbose: bool = True,
    accelerator=None,
    wandb_api_key=None,
):

    # if wb:
    #     if not hasattr(wandb, "api"):
    #         if wandb_api_key is not None:
    #             wandb.login(key=wandb_api_key)
    #         else:
    #             warnings.warn("Must login to wandb with API key.")

    #     config_dict = {}
    #     if INIT_HP is not None:
    #         config_dict.update(INIT_HP)
    #     if MUT_P is not None:
    #         config_dict.update(MUT_P)

    #     if accelerator is not None:
    #         accelerator.wait_for_everyone()
    #         if accelerator.is_main_process:
    #             wandb.init(
    #                 # set the wandb project where this run will be logged
    #                 project="AgileRL",
    #                 name="{}-EvoHPO-{}-{}".format(
    #                     "rnl", algo, datetime.now().strftime("%m%d%Y%H%M%S")
    #                 ),
    #                 # track hyperparameters and run metadata
    #                 config=config_dict,
    #             )
    #         accelerator.wait_for_everyone()
    #     else:
    #         wandb.init(
    #             # set the wandb project where this run will be logged
    #             project="AgileRL",
    #             name="{}-EvoHPO-{}-{}".format(
    #                 "rnl", algo, datetime.now().strftime("%m%d%Y%H%M%S")
    #             ),
    #             # track hyperparameters and run metadata
    #             config=config_dict,
    #         )

    if accelerator is not None:
        accel_temp_models_path = "models/rnl"
        if accelerator.is_main_process:
            if not os.path.exists(accel_temp_models_path):
                os.makedirs(accel_temp_models_path)

    num_envs = env.num_envs

    save_path = (
        checkpoint_path.split(".pt")[0]
        if checkpoint_path is not None
        else "{}-EvoHPO-{}-{}".format(
            "rnl", "rainbow", datetime.now().strftime("%m%d%Y%H%M%S")
        )
    )

    if accelerator is not None:
        # Create dataloader from replay buffer
        replay_dataset = ReplayDataset(memory, pop[0].batch_size)
        replay_dataloader = DataLoader(replay_dataset, batch_size=None)
        replay_dataloader = accelerator.prepare(replay_dataloader)
        sampler = Sampler(
            distributed=True, dataset=replay_dataset, dataloader=replay_dataloader
        )
    else:
        sampler = Sampler(distributed=False, per=per, memory=memory)
        n_step_sampler = Sampler(distributed=False, n_step=True, memory=n_step_memory)

    if accelerator is not None:
        print(f"\nDistributed training on {accelerator.device}...")
    else:
        print("\nTraining...")

    bar_format = "{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]"
    if accelerator is not None:
        pbar = trange(
            max_steps,
            unit="step",
            bar_format=bar_format,
            ascii=True,
            disable=not accelerator.is_local_main_process,
        )
    else:
        pbar = trange(max_steps, unit="step", bar_format=bar_format, ascii=True)

    pop_loss = [[] for _ in pop]
    pop_fitnesses = []
    total_steps = 0
    loss = None
    checkpoint_count = 0

    # Pre-training mutation
    if accelerator is None:
        if mutation is not None:
            pop = mutation.mutation(pop, pre_training_mut=True)

    # RL training loop
    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        if accelerator is not None:
            accelerator.wait_for_everyone()
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
                if one_step_transition:
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
                        experiences = sampler.sample(agent.batch_size, agent.beta)
                        if n_step_memory is not None:
                            n_step_experiences = n_step_sampler.sample(experiences[6])
                            experiences += n_step_experiences
                        loss, idxs, priorities = agent.learn_dqn(experiences)
                        memory.update_priorities(idxs, priorities)

                elif (
                    len(memory) >= agent.batch_size and memory.counter > learning_delay
                ):
                    for _ in range(num_envs // agent.learn_step):
                        # Sample replay buffer
                        # Learn according to agent's RL algorithm
                        experiences = sampler.sample(agent.batch_size, agent.beta)
                        if n_step_memory is not None:
                            n_step_experiences = n_step_sampler.sample(experiences[6])
                            experiences += n_step_experiences
                        loss, idxs, priorities = agent.learn_dqn(experiences)
                        memory.update_priorities(idxs, priorities)

                if loss is not None:
                    losses.append(loss)

                state = next_state

                # pdb.set_trace()

            agent.steps[-1] += steps
            pbar.update(evo_steps // len(pop))

            pop_episode_scores.append(completed_episode_scores)

            if len(losses) > 0:
                if isinstance(losses[-1], tuple):
                    actor_losses, critic_losses = list(zip(*losses))
                    mean_loss = np.mean(
                        [loss for loss in actor_losses if loss is not None]
                    ), np.mean(critic_losses)
                else:
                    mean_loss = np.mean(losses)
                pop_loss[agent_idx].append(mean_loss)

        # Evaluate population
        fitnesses = [
            agent.test(env, max_steps=eval_steps, loop=eval_loop) for agent in pop
        ]
        pop_fitnesses.append(fitnesses)
        mean_scores = [
            (
                np.mean(episode_scores)
                if len(episode_scores) > 0
                else "0 completed episodes"
            )
            for episode_scores in pop_episode_scores
        ]

        # if wb:
        #     wandb_dict = {
        #         "global_step": (
        #             total_steps * accelerator.state.num_processes
        #             if accelerator is not None and accelerator.is_main_process
        #             else total_steps
        #         ),
        #         "train/mean_score": np.mean(
        #             [
        #                 mean_score
        #                 for mean_score in mean_scores
        #                 if not isinstance(mean_score, str)
        #             ]
        #         ),
        #         "eval/mean_fitness": np.mean(fitnesses),
        #         "eval/best_fitness": np.max(fitnesses),
        #     }

        #     # Create the loss dictionaries
        #     if algo in ["RainbowDQN", "DQN"]:
        #         actor_loss_dict = {
        #             f"train/agent_{index}_actor_loss": np.mean(loss[-10:])
        #             for index, loss in enumerate(pop_loss)
        #         }
        #         wandb_dict.update(actor_loss_dict)

        #     if algo in ["DQN", "Rainbow DQN"]:
        #         train_actions_hist = [
        #             freq / sum(train_actions_hist) for freq in train_actions_hist
        #         ]
        #         train_actions_dict = {
        #             f"train/action_{index}": action
        #             for index, action in enumerate(train_actions_hist)
        #         }
        #         wandb_dict.update(train_actions_dict)

        #     if accelerator is not None:
        #         accelerator.wait_for_everyone()
        #         if accelerator.is_main_process:
        #             wandb.log(wandb_dict)
        #         accelerator.wait_for_everyone()
        #     else:
        #         wandb.log(wandb_dict)

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
                # if wb:
                #     wandb.finish()
                return pop, pop_fitnesses

        # Tournament selection and population mutation
        if tournament and mutation is not None:
            if accelerator is not None:
                accelerator.wait_for_everyone()
                for model in pop:
                    model.unwrap_models()
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    elite, pop = tournament.select(pop)
                    pop = mutation.mutation(pop)
                    for pop_i, model in enumerate(pop):
                        model.save_checkpoint(
                            f"{accel_temp_models_path}/rainbow_{pop_i}.pt"
                        )
                accelerator.wait_for_everyone()
                if not accelerator.is_main_process:
                    for pop_i, model in enumerate(pop):
                        model.load_checkpoint(
                            f"{accel_temp_models_path}/rainbow_{pop_i}.pt"
                        )
                accelerator.wait_for_everyone()
                for model in pop:
                    model.wrap_models()
            else:
                elite, pop = tournament.select(pop)
                pop = mutation.mutation(pop)

            if save_elite:
                elite_save_path = (
                    elite_path.split(".pt")[0]
                    if elite_path is not None
                    else "rnl-elite_rainbow"
                )
                elite.save_checkpoint(f"{elite_save_path}.pt")

        if verbose:
            fitness = ["%.2f" % fitness for fitness in fitnesses]
            avg_fitness = ["%.2f" % np.mean(agent.fitness[-5:]) for agent in pop]
            avg_score = ["%.2f" % np.mean(agent.scores[-10:]) for agent in pop]
            agents = [agent.index for agent in pop]
            num_steps = [agent.steps[-1] for agent in pop]
            muts = [agent.mut for agent in pop]
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
                Mutations:\t\t{muts}
                """,
                end="\r",
            )

        # Save model checkpoint
        if checkpoint is not None:
            if pop[0].steps[-1] // checkpoint > checkpoint_count:
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                    for model in pop:
                        model.unwrap_models()
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        for i, agent in enumerate(pop):
                            current_checkpoint_path = (
                                f"{save_path}_{i}.pt"
                                if overwrite_checkpoints
                                else f"{save_path}_{i}_{agent.steps[-1]}.pt"
                            )
                            agent.save_checkpoint(current_checkpoint_path)
                        print("Saved checkpoint.")
                    accelerator.wait_for_everyone()
                    for model in pop:
                        model.wrap_models()
                    accelerator.wait_for_everyone()
                else:
                    for i, agent in enumerate(pop):
                        current_checkpoint_path = (
                            f"{save_path}_{i}.pt"
                            if overwrite_checkpoints
                            else f"{save_path}_{i}_{agent.steps[-1]}.pt"
                        )
                        agent.save_checkpoint(current_checkpoint_path)
                    print("Saved checkpoint.")
                checkpoint_count += 1

    # if wb:
    #     if accelerator is not None:
    #         accelerator.wait_for_everyone()
    #         if accelerator.is_main_process:
    #             wandb.finish()
    #         accelerator.wait_for_everyone()
    #     else:
    #         wandb.finish()

    pbar.close()
    return pop, pop_fitnesses
