import gym
import hydra
from algorithm import Agent
from components import ReplayBuffer
from engine import Collision, GenerateWorld
from environment import Generator, Robot
from hydra.core.config_store import ConfigStore
from models import ModelActor, ModelCritic
from training import Engine, TrainerConfig

import microvault

cs = ConfigStore.instance()
cs.store(name="trainer_config", node=TrainerConfig)


@hydra.main(config_path="microvault/configs", config_name="config", version_base="1.2")
def main(cfg: TrainerConfig):

    engine = Engine(seed=cfg.engine.seed, device=cfg.engine.device)
    replaybuffer = ReplayBuffer(
        buffer_size=cfg.replay_buffer.buffer_size,
        batch_size=cfg.engine.batch_size,
        gamma=cfg.agent.gamma,
        nstep=cfg.agent.nstep,
        state_dim=cfg.environment.state_size,
        action_dim=cfg.environment.action_size,
        epsilon=cfg.replay_buffer.epsilon,
        alpha=cfg.replay_buffer.alpha,
        beta=cfg.replay_buffer.beta,
        beta_increment_per_sampling=cfg.replay_buffer.beta_increment_per_sampling,
        absolute_error_upper=cfg.replay_buffer.absolute_error_upper,
    )
    modelActor = ModelActor(
        state_dim=cfg.environment.state_size,
        action_dim=cfg.environment.action_size,
        max_action=cfg.robot.max_action,
        l1=cfg.network.layers_actor_l1,
        l2=cfg.network.layers_actor_l2,
        device=cfg.engine.device,
        batch_size=cfg.engine.batch_size,
    )
    modelCritic = ModelCritic(
        state_dim=cfg.environment.state_size,
        action_dim=cfg.environment.action_size,
        l1=cfg.network.layers_critic_l1,
        l2=cfg.network.layers_critic_l2,
        device=cfg.engine.device,
        batch_size=cfg.engine.batch_size,
    )
    agent = Agent(
        replayBuffer=replaybuffer,
        modelActor=modelActor,
        modelCritic=modelCritic,
        state_size=cfg.environment.state_size,
        action_size=cfg.environment.action_size,
        max_action=cfg.robot.max_action,
        min_action=cfg.robot.min_action,
        batch_size=cfg.engine.batch_size,
        update_every_step=cfg.agent.update_every_step,
        gamma=cfg.agent.gamma,
        tau=cfg.agent.tau,
        lr_actor=cfg.agent.lr_actor,
        lr_critic=cfg.agent.lr_critic,
        weight_decay=cfg.agent.weight_decay,
        noise=cfg.agent.noise,
        noise_std=cfg.agent.noise_std,
        noise_clip=cfg.agent.noise_clip,
        random_seed=cfg.engine.seed,
        device=cfg.engine.device,
        pretrained=cfg.engine.pretrained,
        nstep=cfg.agent.nstep,
        desired_distance=cfg.noise_layer.desired_distance,
        scalar=cfg.noise_layer.scalar,
        scalar_decay=cfg.noise_layer.scalar_decay,
    )

    generate = Generator(
        collision=Collision(),
        generate=GenerateWorld(),
        grid_lenght=cfg.environment.grid_lenght,
        random=cfg.environment.random,
    )
    robot = Robot(
        collision=Collision(),
        time=cfg.environment.timestep,
        min_radius=cfg.robot.min_radius,
        max_radius=cfg.robot.max_radius,
        max_grid=cfg.environment.grid_lenght,
        wheel_radius=cfg.robot.wheel_radius,
        wheel_base=cfg.robot.wheel_base,
        fov=cfg.robot.fov,
        num_rays=cfg.robot.num_rays,
        max_range=cfg.robot.max_range,
    )

    env = gym.make(
        "microvault/NaviEnv-v0",
        rgb_array=True,
        max_episode=1000,
        robot=robot,
        generator=generate,
    )

    if cfg.engine.train:
        state = env.reset()
        done = False

        for timestep in range(100):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
    else:
        state = env.reset()
        env.render()


if __name__ == "__main__":
    main()
