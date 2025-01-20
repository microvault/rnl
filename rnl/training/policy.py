import warnings
from datetime import datetime

import numpy as np
from tqdm import trange

import wandb


def train_on_policy(
    env,
    env_name,
    algo,
    pop,
    INIT_HP=None,
    MUT_P=None,
    swap_channels=False,
    max_steps=1000000,
    evo_steps=10000,
    eval_steps=None,
    eval_loop=1,
    target=None,
    tournament=None,
    mutation=None,
    checkpoint=None,
    checkpoint_path=None,
    overwrite_checkpoints=False,
    save_elite=False,
    elite_path=None,
    wb=False,
    verbose=True,
    accelerator=None,
    wandb_api_key=None,
):

    assert isinstance(
        algo, str
    ), "'algo' must be the name of the algorithm as a string."
    assert isinstance(max_steps, int), "Number of steps must be an integer."
    assert isinstance(evo_steps, int), "Evolution frequency must be an integer."
    if target is not None:
        assert isinstance(
            target, (float, int)
        ), "Target score must be a float or an integer."
    if checkpoint is not None:
        assert isinstance(checkpoint, int), "Checkpoint must be an integer."
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
        if INIT_HP is not None:
            config_dict.update(INIT_HP)
        if MUT_P is not None:
            config_dict.update(MUT_P)

        wandb.init(
            # set the wandb project where this run will be logged
            project="rnl",
            name="{}-rnl-{}-{}".format(
                env_name, algo, datetime.now().strftime("%m%d%Y%H%M%S")
            ),
            # track hyperparameters and run metadata
            config=config_dict,
        )

    num_envs = env.num_envs

    save_path = (
        checkpoint_path.split(".pt")[0]
        if checkpoint_path is not None
        else "{}-rnl-{}-{}".format(
            env_name, algo, datetime.now().strftime("%m%d%Y%H%M%S")
        )
    )

    print("\nTraining...")
    bar_format = "{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]"

    pbar = trange(
        max_steps,
        unit="step",
        bar_format=bar_format,
        ascii=True,
        dynamic_ncols=True,
    )

    pop_loss = [[] for _ in pop]
    pop_fitnesses = []
    total_steps = 0
    loss = None
    checkpoint_count = 0

    pop_values = [[] for _ in pop]
    pop_log_probs = [[] for _ in pop]

    # Pre-training mutation
    pop = mutation.mutation(pop, pre_training_mut=True)

    # Initialize metrics
    policy_entropies = [[] for _ in pop]
    kl_divergences = [[] for _ in pop]
    residual_variances = [[] for _ in pop]

    # RL training loop
    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        pop_episode_scores = []
        for agent_idx, agent in enumerate(pop):  # Loop through population
            state, info = env.reset()  # Reset environment at start of episode
            scores = np.zeros(num_envs)
            completed_episode_scores = []
            steps = 0

            for _ in range(-(evo_steps // -agent.learn_step)):

                states = []
                actions = []
                log_probs = []
                rewards = []
                dones = []
                values = []
                truncs = []

                learn_steps = 0

                for idx_step in range(-(agent.learn_step // -num_envs)):
                    # Get next action from agent
                    action, log_prob, _, value = agent.get_action(
                        state, action_mask=None
                    )

                    next_state, reward, done, trunc, info = env.step(action)

                    total_steps += num_envs
                    steps += num_envs
                    learn_steps += num_envs

                    states.append(state)
                    actions.append(action)
                    log_probs.append(log_prob)
                    rewards.append(reward)
                    dones.append(done)
                    values.append(value)
                    truncs.append(trunc)

                    state = next_state
                    scores += np.array(reward)

                    for idx, (d, t) in enumerate(zip(done, trunc)):
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
                # Learn according to agent's RL algorithm
                loss, metrics = agent.learn(experiences)
                pop_loss[agent_idx].append(loss)

                if verbose:
                    pop_values[agent_idx].extend(values)
                    pop_log_probs[agent_idx].extend(log_probs)

                    # Store additional metrics
                    policy_entropies[agent_idx].append(metrics.get("entropy", 0))
                    kl_divergences[agent_idx].append(metrics.get("kl_div", 0))
                    residual_variances[agent_idx].append(metrics.get("residual_var", 0))


            agent.steps[-1] += steps
            pop_episode_scores.append(completed_episode_scores)

        # Evaluate population
        fitnesses = [
            agent.test(
                env, swap_channels=swap_channels, max_steps=eval_steps, loop=eval_loop
            )
            for agent in pop
        ]

        best_agent_idx = int(np.argmax(fitnesses))
        best_agent = pop[best_agent_idx]
        video_result = best_agent.test_with_animation()

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
            mean_values = [np.mean(v) if len(v) > 0 else 0 for v in pop_values]
            mean_log_probs = [np.mean(lp) if len(lp) > 0 else 0 for lp in pop_log_probs]
            mean_entropies = [
                np.mean(ent) if len(ent) > 0 else 0 for ent in policy_entropies
            ]
            mean_kl_div = [np.mean(kl) if len(kl) > 0 else 0 for kl in kl_divergences]
            mean_residual_variance = [
                            np.mean(rv) if len(rv) > 0 else 0 for rv in residual_variances
                        ]
            wandb_dict = {
                "global_step": (total_steps),
                "train/mean_score": np.mean(
                    [
                        mean_score
                        for mean_score in mean_scores
                        if not isinstance(mean_score, str)
                    ]
                ),
                "eval/mean_fitness": np.mean(fitnesses),
                "eval/best_fitness": np.max(fitnesses),
                "train/mean_values": np.mean(mean_values),
                "train/mean_log_probs": np.mean(mean_log_probs),
                "train/mean_entropy": np.mean(mean_entropies),
                "train/mean_kl_div": np.mean(mean_kl_div),
                "train/mean_residual_variance": np.mean(mean_residual_variance),
                "train/video": video_result,
            }

            agent_loss_dict = {
                f"train/agent_{index}_loss": np.mean(loss_[-10:])
                for index, loss_ in enumerate(pop_loss)
            }
            wandb_dict.update(agent_loss_dict)

            wandb.log(wandb_dict)

            # Reset metrics for next iteration
            pop_values = [[] for _ in pop]
            pop_log_probs = [[] for _ in pop]
            policy_entropies = [[] for _ in pop]
            kl_divergences = [[] for _ in pop]
            residual_variances = [[] for _ in pop]

        # Update step counter
        for agent in pop:
            agent.steps.append(agent.steps[-1])

        # Early stop if consistently reaches target
        if target is not None:
            print("Target is not None.")
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
            mean_values_verbose = [np.mean(v) if len(v) > 0 else 0 for v in pop_values]
            mean_log_probs_verbose = [
                np.mean(lp) if len(lp) > 0 else 0 for lp in pop_log_probs
            ]
            mean_entropies_verbose = [
                np.mean(ent) if len(ent) > 0 else 0 for ent in policy_entropies
            ]
            mean_kl_div_verbose = [
                np.mean(kl) if len(kl) > 0 else 0 for kl in kl_divergences
            ]
            mean_residual_variance_verbose = [
                        np.mean(rv) if len(rv) > 0 else 0 for rv in residual_variances
                    ]
            pbar.update(0)

            # Formatar os valores para melhor visualização
            mean_values_formatted = [f"{mv:.4f}" for mv in mean_values_verbose]
            mean_log_probs_formatted = [f"{mlp:.4f}" for mlp in mean_log_probs_verbose]
            mean_entropies_formatted = [f"{me:.4f}" for me in mean_entropies_verbose]
            mean_kl_div_formatted = [f"{mkld:.4f}" for mkld in mean_kl_div_verbose]
            mean_residual_variance_formatted = [
                        f"{mrv:.4f}" for mrv in mean_residual_variance_verbose
                    ]

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
                Mean Values:\t\t{mean_values_formatted}
                Mean Log Probs:\t{mean_log_probs_formatted}
                Mean Entropy:\t\t{mean_entropies_formatted}
                Mean KL Div:\t\t{mean_kl_div_formatted}
                Mean Residual Var:\t{mean_residual_variance_formatted}
                """,
                end="\r",
            )

            # Reset pop_values e pop_log_probs para a próxima iteração
            # Reset metrics for next iteration
            pop_values = [[] for _ in pop]
            pop_log_probs = [[] for _ in pop]
            policy_entropies = [[] for _ in pop]
            kl_divergences = [[] for _ in pop]
            residual_variances = [[] for _ in pop]

        # Save model checkpoint
        if checkpoint is not None:
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
