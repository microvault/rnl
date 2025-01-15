import numpy as np
import torch
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import create_population
from tqdm import trange
import wandb
import os
import warnings
from datetime import datetime

from rnl.configs.config import (
    AgentConfig,
    EnvConfig,
    HPOConfig,
    NetworkConfig,
    RenderConfig,
    RobotConfig,
    SensorConfig,
    TrainerConfig,
)
from rnl.environment.environment_navigation import NaviEnv
from rnl.training.utils import make_vect_envs
from agilerl.components.replay_data import ReplayDataset
from agilerl.components.sampler import Sampler
from torch.utils.data import DataLoader


def training(
    agent_config: AgentConfig,
    trainer_config: TrainerConfig,
    hpo_config: HPOConfig,
    network_config: NetworkConfig,
    robot_config: RobotConfig,
    sensor_config: SensorConfig,
    env_config: EnvConfig,
    render_config: RenderConfig,
    pretrained_model: bool,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_dict = {
        "Agent Config": agent_config.__dict__,
        "Trainer Config": trainer_config.__dict__,
        "HPO Config": hpo_config.__dict__,
        "Network Config": network_config.__dict__,
    }

    for config_name, config_values in config_dict.items():
        print(f"\n#------ {config_name} ----#")
        max_key_length = max(len(key) for key in config_values.keys())
        for key, value in config_values.items():
            print(f"{key.ljust(max_key_length)} : {value}")

    INIT_PARAM = {
        "ENV_NAME": "rnl",  # Gym environment name
        "ALGO": "Rainbow DQN",  # Algorithm
        "DOUBLE": True,  # Use double Q-learning
        "CHANNELS_LAST": False,  # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "BATCH_SIZE": trainer_config.batch_size,  # Batch size
        "LR": trainer_config.lr,  # Learning rate
        "MAX_STEPS": trainer_config.max_steps,  # Max no. steps
        "TARGET_SCORE": trainer_config.target_score,  # Early training stop at avg score of last 100 episodes
        "GAMMA": agent_config.gamma,  # Discount factor
        "TAU": agent_config.tau,  # For soft update of target parameters
        "BETA": agent_config.beta,  # PER beta
        "PRIOR_EPS": agent_config.prior_eps,  # PER epsilon
        "MEMORY_SIZE": agent_config.memory_size,  # Max memory buffer size
        "LEARN_STEP": trainer_config.learn_step,  # Learning frequency
        "TOURN_SIZE": hpo_config.tourn_size,  # Tournament size
        "ELITISM": hpo_config.elitism,  # Elitism in tournament selection
        "POP_SIZE": hpo_config.population_size,  # Population size
        "EVO_STEPS": hpo_config.evo_steps,  # Evolution frequency
        "EVAL_STEPS": hpo_config.eval_steps,  # Evaluation steps
        "EVAL_LOOP": hpo_config.eval_loop,  # Evaluation episodes
        "LEARNING_DELAY": trainer_config.learning_delay,  # Steps before starting learning
        "WANDB": trainer_config.use_wandb,  # Log with Weights and Biases
        "WANDB_API_KEY": trainer_config.wandb_api_key,
        "CHECKPOINT": trainer_config.checkpoint,  # Checkpoint frequency
        "CHECKPOINT_PATH": trainer_config.checkpoint_path,  # Checkpoint path
        "OVERWRITE_CHECKPOINTS": trainer_config.overwrite_checkpoints,
        "SAVE_ELITE": hpo_config.save_elite,  # Save elite agent
        "ELITE_PATH": hpo_config.elite_path,  # Elite agent path
        "ACCELERATOR": None,
        "VERBOSE": True,
        "TORCH_COMPILER": True,
        "EPS_START": trainer_config.eps_start,
        "EPS_END": trainer_config.eps_end,
        "EPS_DECAY": trainer_config.eps_decay,
        "N_STEP": agent_config.n_step,
        "PER": agent_config.per,
        "N_STEP_MEMORY": trainer_config.n_step_memory,
        "NUM_ATOMS": agent_config.num_atoms,
        "V_MIN": agent_config.v_min,
        "V_MAX": agent_config.v_max,
    }

    NET_CONFIG = {
        "arch": network_config.arch,  # Network architecture
        "hidden_size": network_config.hidden_size,  # Actor hidden size
        "mlp_activation": network_config.mlp_activation,
        "mlp_output_activation": network_config.mlp_output_activation,
        "min_hidden_layers": network_config.min_hidden_layers,
        "max_hidden_layers": network_config.max_hidden_layers,
        "min_mlp_nodes": network_config.min_mlp_nodes,
        "max_mlp_nodes": network_config.max_mlp_nodes,
    }

    MUTATION_PARAMS = {
        "NO_MUT": hpo_config.no_mutation,  # No mutation
        "ARCH_MUT": hpo_config.arch_mutation,  # Architecture mutation
        "NEW_LAYER": hpo_config.new_layer,  # New layer mutation
        "PARAMS_MUT": hpo_config.param_mutation,  # Network parameters mutation
        "ACT_MUT": hpo_config.active_mutation,  # Activation layer mutation
        "RL_HP_MUT": hpo_config.hp_mutation,  # Learning HP mutation
        "RL_HP_SELECTION": hpo_config.hp_mutation_selection,
        "MUT_SD": hpo_config.mutation_strength,  # Mutation strength
        "MIN_LR": hpo_config.min_lr,
        "MAX_LR": hpo_config.max_lr,
        "MIN_LEARN_STEP": hpo_config.min_learn_step,
        "MAX_LEARN_STEP": hpo_config.max_learn_step,
        "MIN_BATCH_SIZE": hpo_config.min_batch_size,
        "MAX_BATCH_SIZE": hpo_config.max_batch_size,
        "AGENTS_IDS": None,
        "MUTATE_ELITE": hpo_config.mutate_elite,
        "RAND_SEED": hpo_config.rand_seed,  # Random seed
        "ACTIVATION": hpo_config.activation,  # Activation functions to choose from
    }

    num_envs = trainer_config.num_envs

    env = make_vect_envs(
        num_envs=num_envs,
        robot_config=robot_config,
        sensor_config=sensor_config,
        env_config=env_config,
        render_config=render_config,
    )
    state_dim = env.single_observation_space.shape
    action_dim = env.single_action_space.n

    pop = create_population(
        algo=INIT_PARAM["ALGO"],  # Algorithm
        state_dim=state_dim,  # State dimension
        action_dim=action_dim,  # Action dimension
        one_hot=False,  # One-hot encoding
        net_config=NET_CONFIG,  # Network configuration
        INIT_HP=INIT_PARAM,  # Initial hyperparameters
        population_size=INIT_PARAM["POP_SIZE"],  # Population size
        num_envs=num_envs,  # Number of vectorized environments
        device=device,
        accelerator=INIT_PARAM["ACCELERATOR"],
        torch_compiler=INIT_PARAM["TORCH_COMPILER"],
    )

    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(
        memory_size=INIT_PARAM["MEMORY_SIZE"],  # Max replay buffer size
        field_names=field_names,  # Field names to store in memory
        device=device,
    )

    tournament = TournamentSelection(
        tournament_size=INIT_PARAM["TOURN_SIZE"],  # Tournament selection size
        elitism=INIT_PARAM["ELITISM"],  # Elitism in tournament selection
        population_size=INIT_PARAM["POP_SIZE"],  # Population size
        eval_loop=INIT_PARAM["EVAL_LOOP"],  # Evaluate using last N fitness scores
    )

    mutation = Mutations(
        algo=INIT_PARAM["ALGO"],  # Algorithm
        no_mutation=MUTATION_PARAMS["NO_MUT"],  # No mutation
        architecture=MUTATION_PARAMS["ARCH_MUT"],  # Architecture mutation
        new_layer_prob=MUTATION_PARAMS["NEW_LAYER"],  # New layer mutation
        parameters=MUTATION_PARAMS["PARAMS_MUT"],  # Network parameters mutation
        activation=MUTATION_PARAMS["ACT_MUT"],  # Activation layer mutation
        rl_hp=MUTATION_PARAMS["RL_HP_MUT"],  # Learning HP mutation
        rl_hp_selection=MUTATION_PARAMS[
            "RL_HP_SELECTION"
        ],  # Learning HPs to choose from
        mutation_sd=MUTATION_PARAMS["MUT_SD"],  # Mutation strength
        activation_selection=MUTATION_PARAMS[
            "ACTIVATION"
        ],  # Activation functions to choose from
        min_lr=MUTATION_PARAMS["MIN_LR"],
        max_lr=MUTATION_PARAMS["MAX_LR"],
        min_learn_step=MUTATION_PARAMS["MIN_LEARN_STEP"],
        max_learn_step=MUTATION_PARAMS["MAX_LEARN_STEP"],
        min_batch_size=MUTATION_PARAMS["MIN_BATCH_SIZE"],
        max_batch_size=MUTATION_PARAMS["MAX_BATCH_SIZE"],
        agent_ids=MUTATION_PARAMS["AGENTS_IDS"],
        mutate_elite=MUTATION_PARAMS["MUTATE_ELITE"],
        arch=NET_CONFIG["arch"],  # Network architecture
        rand_seed=MUTATION_PARAMS["RAND_SEED"],  # Random seed
        device=device,
        accelerator=INIT_PARAM["ACCELERATOR"],
    )

    algo=INIT_PARAM["ALGO"]
    env_name=INIT_PARAM["ENV_NAME"]
    algo=INIT_PARAM["ALGO"]
    memory=memory
    swap_channels=INIT_PARAM[
        "CHANNELS_LAST"
    ]
    n_step=False
    per=False
    n_step_memory=None
    max_steps=INIT_PARAM["MAX_STEPS"]
    evo_steps=INIT_PARAM["EVO_STEPS"]
    eval_steps=INIT_PARAM["EVAL_STEPS"]
    eval_loop=INIT_PARAM["EVAL_LOOP"]
    learning_delay=INIT_PARAM["LEARNING_DELAY"]
    eps_start=INIT_PARAM["EPS_START"]
    eps_end=INIT_PARAM["EPS_END"]
    eps_decay=INIT_PARAM["EPS_DECAY"]
    target=INIT_PARAM["TARGET_SCORE"]
    wb=INIT_PARAM["WANDB"]
    checkpoint=INIT_PARAM["CHECKPOINT"]
    checkpoint_path=INIT_PARAM["CHECKPOINT_PATH"]
    save_elite=INIT_PARAM["SAVE_ELITE"]
    elite_path=INIT_PARAM["ELITE_PATH"]
    verbose=INIT_PARAM["VERBOSE"]
    accelerator=INIT_PARAM["ACCELERATOR"]
    wandb_api_key=INIT_PARAM["WANDB_API_KEY"]
    overwrite_checkpoints = INIT_PARAM["OVERWRITE_CHECKPOINTS"]


    assert isinstance(
        algo, str
    ), "'algo' must be the name of the algorithm as a string."
    assert isinstance(max_steps, int), "Number of steps must be an integer."
    assert isinstance(evo_steps, int), "Evolution frequency must be an integer."
    assert isinstance(eps_start, float), "Starting epsilon must be a float."
    assert isinstance(eps_end, float), "Final value of epsilone must be a float."
    assert isinstance(eps_decay, float), "Epsilon decay rate must be a float."
    if target is not None:
        assert isinstance(
            target, (float, int)
        ), "Target score must be a float or an integer."
    if checkpoint is not None:
        assert isinstance(checkpoint, int), "Checkpoint must be an integer."
    assert isinstance(n_step, bool), "'n_step' must be a boolean."
    assert isinstance(per, bool), "'per' must be a boolean."
    assert isinstance(
        wb, bool
    ), "'wb' must be a boolean flag, indicating whether to record run with W&B"
    assert isinstance(verbose, bool), "Verbose must be a boolean."
    if save_elite is False and elite_path is not None:
        warnings.warn(
            "'save_elite' set to False but 'elite_path' has been defined, elite will not\
                      be saved unless 'save_elite' is set to True."
        )
    if checkpoint is None and checkpoint_path is not None:
        warnings.warn(
            "'checkpoint' set to None but 'checkpoint_path' has been defined, checkpoint will not\
                      be saved unless 'checkpoint' is defined."
        )

    if wb:
        if not hasattr(wandb, "api"):
            if wandb_api_key is not None:
                wandb.login(key=wandb_api_key)
            else:
                warnings.warn("Must login to wandb with API key.")

        config_dict = {}
        if INIT_PARAM is not None:
            config_dict.update(INIT_PARAM)
        if INIT_PARAM is not None:
            config_dict.update(MUTATION_PARAMS)

        if accelerator is not None:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="AgileRL",
                    name="EvoHPO-{}".format(datetime.now().strftime("%m%d%Y%H%M%S")
                    ),
                    # track hyperparameters and run metadata
                    config=config_dict,
                )
            accelerator.wait_for_everyone()
        else:
            wandb.init(
                # set the wandb project where this run will be logged
                project="AgileRL",
                name="EvoHPO-{}".format(datetime.now().strftime("%m%d%Y%H%M%S")
                ),
                # track hyperparameters and run metadata
                config=config_dict,
            )

    if accelerator is not None:
        accel_temp_models_path = "models/rnl"
        if accelerator.is_main_process:
            if not os.path.exists(accel_temp_models_path):
                os.makedirs(accel_temp_models_path)

    # Detect if environment is vectorised
    if hasattr(env, "num_envs"):
        num_envs = env.num_envs
        is_vectorised = True
    else:
        is_vectorised = False
        num_envs = 1

    save_path = (
        checkpoint_path.split(".pt")[0]
        if checkpoint_path is not None
        else "EvoHPO-{}".format(datetime.now().strftime("%m%d%Y%H%M%S")
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
        if n_step_memory is not None:
            n_step_sampler = Sampler(
                distributed=False, n_step=True, memory=n_step_memory
            )

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
        for agent_idx, agent in enumerate(pop):  # Loop through population
            state, info = env.reset()  # Reset environment at start of episode
            scores = np.zeros(num_envs)
            completed_episode_scores, losses = [], []
            steps = 0

            if algo in ["DQN", "Rainbow DQN"]:
                train_actions_hist = [0] * agent.action_dim

            if algo in ["DQN"]:
                epsilon = eps_start

            for idx_step in range(evo_steps // num_envs):
                if swap_channels:
                    state = np.moveaxis(state, [-1], [-3])

                # Get next action from agent
                if algo in ["DQN"]:
                    action_mask = info.get("action_mask", None)
                    action = agent.get_action(state, epsilon, action_mask=action_mask)
                    # Decay epsilon for exploration
                    epsilon = max(eps_end, epsilon * eps_decay)
                if algo in ["Rainbow DQN"]:
                    action_mask = info.get("action_mask", None)
                    action = agent.get_action(state, action_mask=action_mask)
                else:
                    action = agent.get_action(state)

                if algo in ["DQN", "Rainbow DQN"]:
                    for a in action:
                        if not isinstance(a, int):
                            a = int(a)
                        train_actions_hist[a] += 1

                if not is_vectorised:
                    action = action[0]

                # Act in environment
                next_state, reward, done, trunc, info = env.step(action)
                scores += np.array(reward)

                if not is_vectorised:
                    done = [done]
                    trunc = [trunc]

                reset_noise_indices = []
                for idx, (d, t) in enumerate(zip(done, trunc)):
                    if d or t:
                        completed_episode_scores.append(scores[idx])
                        agent.scores.append(scores[idx])
                        scores[idx] = 0
                        reset_noise_indices.append(idx)
                if agent.algo in ["DDPG", "TD3"]:
                    agent.reset_action_noise(reset_noise_indices)

                total_steps += num_envs
                steps += num_envs

                # Save experience to replay buffer
                if n_step_memory is not None:
                    if swap_channels:
                        one_step_transition = n_step_memory.save_to_memory_vect_envs(
                            state,
                            action,
                            reward,
                            np.moveaxis(next_state, [-1], [-3]),
                            done,
                        )
                    else:
                        one_step_transition = n_step_memory.save_to_memory_vect_envs(
                            state,
                            action,
                            reward,
                            next_state,
                            done,
                        )
                    if one_step_transition:
                        memory.save_to_memory_vect_envs(*one_step_transition)
                else:
                    if swap_channels:
                        memory.save_to_memory(
                            state,
                            action,
                            reward,
                            np.moveaxis(next_state, [-1], [-3]),
                            done,
                            is_vectorised=is_vectorised,
                        )
                    else:
                        memory.save_to_memory(
                            state,
                            action,
                            reward,
                            next_state,
                            done,
                            is_vectorised=is_vectorised,
                        )

                if per:
                    fraction = min(
                        ((agent.steps[-1] + idx_step + 1) * num_envs / max_steps), 1.0
                    )
                    agent.beta += fraction * (1.0 - agent.beta)

                # Learn according to learning frequency
                # Handle learn_step > num_envs
                if agent.learn_step > num_envs:
                    learn_step = agent.learn_step // num_envs
                    if (
                        idx_step % learn_step == 0
                        and len(memory) >= agent.batch_size
                        and memory.counter > learning_delay
                    ):
                        if per:
                            experiences = sampler.sample(agent.batch_size, agent.beta)
                            if n_step_memory is not None:
                                n_step_experiences = n_step_sampler.sample(
                                    experiences[6]
                                )
                                experiences += n_step_experiences
                            loss, idxs, priorities = agent.learn(
                                experiences, n_step=n_step, per=per
                            )
                            memory.update_priorities(idxs, priorities)
                        else:
                            experiences = sampler.sample(
                                agent.batch_size,
                                return_idx=True if n_step_memory is not None else False,
                            )
                            if n_step_memory is not None:
                                n_step_experiences = n_step_sampler.sample(
                                    experiences[5]
                                )
                                experiences += n_step_experiences
                                loss, *_ = agent.learn(experiences, n_step=n_step)
                            else:
                                loss = agent.learn(experiences)
                                if algo == "Rainbow DQN":
                                    loss, *_ = loss

                elif (
                    len(memory) >= agent.batch_size and memory.counter > learning_delay
                ):
                    for _ in range(num_envs // agent.learn_step):
                        # Sample replay buffer
                        # Learn according to agent's RL algorithm
                        if per:
                            experiences = sampler.sample(agent.batch_size, agent.beta)
                            if n_step_memory is not None:
                                n_step_experiences = n_step_sampler.sample(
                                    experiences[6]
                                )
                                experiences += n_step_experiences
                            loss, idxs, priorities = agent.learn(
                                experiences, n_step=n_step, per=per
                            )
                            memory.update_priorities(idxs, priorities)
                        else:
                            experiences = sampler.sample(
                                agent.batch_size,
                                return_idx=True if n_step_memory is not None else False,
                            )
                            if n_step_memory is not None:
                                n_step_experiences = n_step_sampler.sample(
                                    experiences[5]
                                )
                                experiences += n_step_experiences
                                loss, *_ = agent.learn(experiences, n_step=n_step)
                            else:
                                loss = agent.learn(experiences)
                                if algo == "Rainbow DQN":
                                    loss, *_ = loss

                if loss is not None:
                    losses.append(loss)

                state = next_state

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

        if algo in ["DQN"]:
            # Reset epsilon start to final epsilon value of this epoch
            eps_start = epsilon

        # Evaluate population
        fitnesses = [
            agent.test(
                env, swap_channels=swap_channels, max_steps=eval_steps, loop=eval_loop
            )
            for agent in pop
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

        if wb:
            wandb_dict = {
                "global_step": (
                    total_steps * accelerator.state.num_processes
                    if accelerator is not None and accelerator.is_main_process
                    else total_steps
                ),
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
            if algo in ["Rainbow DQN", "DQN"]:
                actor_loss_dict = {
                    f"train/agent_{index}_actor_loss": np.mean(loss[-10:])
                    for index, loss in enumerate(pop_loss)
                }
                wandb_dict.update(actor_loss_dict)
            elif algo in ["TD3", "DDPG"]:
                actor_loss_dict = {
                    f"train/agent_{index}_actor_loss": np.mean(
                        list(zip(*loss_list))[0][-10:]
                    )
                    for index, loss_list in enumerate(pop_loss)
                }
                critic_loss_dict = {
                    f"train/agent_{index}_critic_loss": np.mean(
                        list(zip(*loss_list))[-1][-10:]
                    )
                    for index, loss_list in enumerate(pop_loss)
                }
                wandb_dict.update(actor_loss_dict)
                wandb_dict.update(critic_loss_dict)

            if algo in ["DQN", "Rainbow DQN"]:
                train_actions_hist = [
                    freq / sum(train_actions_hist) for freq in train_actions_hist
                ]
                train_actions_dict = {
                    f"train/action_{index}": action
                    for index, action in enumerate(train_actions_hist)
                }
                wandb_dict.update(train_actions_dict)

            if accelerator is not None:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    wandb.log(wandb_dict)
                accelerator.wait_for_everyone()
            else:
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
                            f"{accel_temp_models_path}/{algo}_{pop_i}.pt"
                        )
                accelerator.wait_for_everyone()
                if not accelerator.is_main_process:
                    for pop_i, model in enumerate(pop):
                        model.load_checkpoint(
                            f"{accel_temp_models_path}/{algo}_{pop_i}.pt"
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
                    else f"{env_name}-elite_{algo}"
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

    if wb:
        if accelerator is not None:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                wandb.finish()
            accelerator.wait_for_everyone()
        else:
            wandb.finish()

    pbar.close()


def inference(
    robot_config: RobotConfig,
    sensor_config: SensorConfig,
    env_config: EnvConfig,
    render_config: RenderConfig,
    pretrained_model=False,
):

    text = [
        r"+--------------------+",
        r" ____  _   _ _",
        r"|  _ \| \ | | |",
        r"| |_) |  \| | |",
        r"|  _ <| |\  | |___",
        r"|_| \_\_| \_|_____|",
        r"+--------------------+",
    ]

    for line in text:
        print(line)

    config_dict = {
        "Robot Config": robot_config.__dict__,
        "Sensor Config": sensor_config.__dict__,
        "Env Config": env_config.__dict__,
        "Render Config": render_config.__dict__,
    }

    for config_name, config_values in config_dict.items():
        print(f"\n#------ {config_name} ----#")
        max_key_length = max(len(key) for key in config_values.keys())
        for key, value in config_values.items():
            print(f"{key.ljust(max_key_length)} : {value}")

    env = NaviEnv(
        robot_config,
        sensor_config,
        env_config,
        render_config,
        pretrained_model=False,
        use_render=True,
    )

    env.reset()
    env.render()
