import json

from google import genai
from google.genai import types


class LLMTrainingEvaluator:
    """
    Class to evaluate training metrics (average, std deviation, etc.)
    along with user input. Decides if retraining is necessary.
    """

    def __init__(self, evaluator_api_key: str):
        self.evaluator_api_key = evaluator_api_key

    def build_evaluation_prompt(
        self, context_json: dict, stats: dict, context: dict
    ) -> str:
        """
        Creates a prompt describing the metrics and desired behavior.
        """
        stats_info = json.dumps(stats, indent=2)

        base_info = json.dumps(context_json, indent=2)

        context_info = json.dumps(context, indent=2)

        code_str = """
        class NaviEnv(gym.Env):
            def __init__(
                self,
                robot_config: RobotConfig,
                sensor_config: SensorConfig,
                env_config: EnvConfig,
                render_config: RenderConfig,
                use_render: bool,
                actions_cfg: ActionsConfig,
                reward_cfg: RewardConfig,
                mode: str,
            ):
                super().__init__()
                self.max_num_rays = sensor_config.num_rays
                state_size = self.max_num_rays + 5
                self.action_space = spaces.Discrete(3)
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32
                )
                self.robot = Robot(robot_config)
                self.space = self.robot.create_space()
                self.body = self.robot.create_robot(space=self.space)

                self.actions_config = actions_cfg
                self.min_lr = robot_config.vel_linear[0]
                self.max_lr = robot_config.vel_linear[1]
                self.min_vr = robot_config.vel_angular[0]
                self.max_vr = robot_config.vel_angular[1]

                self.reward_config = reward_cfg

                self.mode: str = mode
                self.grid_length = 2
                self.poly = None
                self.infos_list = []

                if "hard" in self.mode:
                    self.grid_lengt = 0
                    self.generator = Generator(mode=self.mode)
                    self.new_map_path, self.segments, self.poly = self.generator.world(
                        self.grid_lengt
                    )

                if "medium" in self.mode:
                    self.create_world = CreateWorld(
                        folder=env_config.folder_map,
                        name=env_config.name_map,
                    )

                    self.new_map_path, self.segments, self.poly = self.create_world.world(
                        mode=self.mode
                    )

                elif "easy" in self.mode:
                    if self.mode in ("easy-00", "easy-01", "easy-02"):
                        self.grid_length = 2

                        self.generator = Generator(mode=self.mode)
                        self.new_map_path, self.segments, self.poly = self.generator.world(
                            self.grid_length
                        )
                    elif self.mode == "easy-03":
                        self.grid_length = 5
                        self.generator = Generator(mode=self.mode)
                        self.new_map_path, self.segments, self.poly = self.generator.world(
                            self.grid_length
                        )
                    elif self.mode == "easy-05":
                        self.grid_length = 10
                        self.generator = Generator(mode=self.mode)
                        self.new_map_path, self.segments, self.poly = self.generator.world(
                            self.grid_length
                        )
                    else:
                        self.generator = Generator(mode=self.mode)
                        self.grid_length = 10
                        self.new_map_path, self.segments, self.poly = self.generator.world(
                            self.grid_length
                        )

                self.sensor = SensorRobot(sensor_config, self.segments)

                # ------------ Normalization ------------ #
                self.scaler_lidar = MinMaxScaler(feature_range=(0, 1))
                self.scaler_dist = MinMaxScaler(feature_range=(0, 1))
                self.scaler_alpha = MinMaxScaler(feature_range=(0, 1))

                max_lidar, min_lidar = sensor_config.max_range, sensor_config.min_range
                self.scaler_lidar.fit(
                    np.array(
                        [
                            [min_lidar] * self.max_num_rays,
                            [max_lidar] * self.max_num_rays,
                        ]
                    )
                )
                self.use_render = use_render
                self.max_dist = compute_polygon_diameter(self.poly) * 0.8 # fator
                self.min_dist = 0.0 # robot_config.threshold
                self.scaler_dist.fit(np.array([[self.min_dist], [self.max_dist]]))

                self.min_alpha, self.max_alpha = 0.0, 3.5 * 0.89
                self.scaler_alpha.fit(np.array([[self.min_alpha], [self.max_alpha]]))
                # -- Environmental parameters -- #
                self.max_lidar = sensor_config.max_range
                self.pretrained_model = robot_config.path_model
                self.max_timestep = env_config.timestep
                self.threshold = robot_config.threshold
                self.collision = robot_config.collision
                self.controller = render_config.controller

                # -- Local Variables -- #

                self.timestep: int = 0
                self.target_x: float = 0.0
                self.target_y: float = 0.0
                self.last_position_x: float = 0.0
                self.last_position_y: float = 0.0
                self.last_theta: float = 0.0
                self.vl: float = 0.01
                self.vr: float = 0.01
                self.action: int = 0
                self.initial_distance: float = 0.0
                self.scalar = env_config.scalar
                self.current_fraction: float = 0.0
                self.debug = render_config.debug
                self.plot = render_config.plot
                self.current_rays = sensor_config.num_rays
                self.lidar_angle = np.linspace(0, 2 * np.pi, self.current_rays)
                self.measurement = np.zeros(self.current_rays)
                self.last_states = np.zeros(state_size)

                self.model = None
                if self.pretrained_model != "None":
                    self.model = PPO.load(robot_config.path_model)

            def step(self, action):
                vl = 0.0
                vr = 0.0

                if action == 0:
                    vl = 0.10 * self.scalar
                    vr = 0.0
                elif action == 1:
                    vl = 0.08 * self.scalar
                    vr = -0.36 * self.scalar
                elif action == 2:
                    vl = 0.08 * self.scalar
                    vr = 0.36 * self.scalar

                self.robot.move_robot(self.space, self.body, vl, vr)

                x, y, theta = (
                    self.body.position.x,
                    self.body.position.y,
                    self.body.angle,
                )

                intersections, lidar_measurements = self.sensor.sensor(
                    x=x, y=y, theta=theta, max_range=self.max_lidar
                )

                dist = distance_to_goal(x, y, self.target_x, self.target_y, self.max_dist)

                alpha = angle_to_goal(
                    self.body.position.x,
                    self.body.position.y,
                    self.body.angle,
                    self.target_x,
                    self.target_y,
                    self.max_alpha
                )

                collision_array, laser = min_laser(lidar_measurements, self.collision)
                collision = bool(np.any(collision_array))

                padded_lidar = np.zeros((self.max_num_rays,), dtype=np.float32)
                padded_lidar[: self.current_rays] = lidar_measurements[: self.current_rays]

                lidar_norm = self.scaler_lidar.transform(padded_lidar.reshape(1, -1)).flatten()
                dist_norm = self.scaler_dist.transform(np.array(dist).reshape(1, -1)).flatten()
                alpha_norm = self.scaler_alpha.transform(
                    np.array(alpha).reshape(1, -1)
                ).flatten()

                action_one_hot = np.eye(3)[action]

                states = np.concatenate(
                    (
                        np.array(lidar_norm, dtype=np.float32),
                        np.array(action_one_hot, dtype=np.int16),
                        np.array(dist_norm, dtype=np.float32),
                        np.array(alpha_norm, dtype=np.float32),
                    )
                )

                (
                    collision_score,
                    orientation_score,
                    progress_score,
                    time_score,
                    obstacle,
                    done,
                ) = self.reward_config.get_reward(
                    lidar_measurements,
                    poly=self.poly,
                    position_x=x,
                    position_y=y,
                    initial_distance=self.initial_distance,
                    current_distance=dist_norm[0],
                    collision=collision,
                    alpha=alpha_norm[0],
                    step=self.timestep,
                    threshold=self.threshold,
                    threshold_collision=self.collision,
                    min_distance=self.min_dist,
                    max_distance=self.max_dist,
                )

                reward = (
                    collision_score + orientation_score + progress_score + time_score + obstacle
                )
                self.last_states = states

                self.space.step(1 / 60)

                self.timestep += 1

                truncated = self.timestep >= self.max_timestep

                if self.debug:

                    info = {
                        "obstacle_score": obstacle,
                        "orientation_score": orientation_score,
                        "progress_score": progress_score,
                        "time_score": time_score,
                        "action": float(action),
                        "dist": float(dist_norm[0]),
                        "alpha": float(alpha_norm[0]),
                        "min_lidar": float(min(lidar_norm)),
                        "max_lidar": float(max(lidar_norm)),
                    }
                    info = clean_info(info)
                    self.infos_list.append(info)
                    return states, reward, done, truncated, info

                else:
                    return states, reward, done, truncated, {}

            def get_infos(self):
                infos = self.infos_list.copy()
                self.infos_list.clear()
                return infos

            def update_strategy(self, new_action_type, new_reward_type, new_params):
                self.actions_config = new_action_type

                self.reward_config = RewardConfig(
                    reward_type=new_reward_type,
                    params=new_params,
                    description=f"Reward configurado para {new_reward_type}",
                )

            def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
                super().reset(seed=seed, options=options)

                self.timestep = 0

                try:
                    if self.mode == "easy-00":
                        self.new_map_path, self.segments, self.poly = self.generator.world(
                            self.grid_length
                        )
                        targets = np.array([[0.35, 0.35], [1.8, 1.8]])
                        choice = targets[np.random.randint(0, len(targets))]
                        self.target_x, self.target_y = choice[0], choice[1]
                        x, y = 1.07, 1.07

                        self.sensor.update_map(self.segments)

                        theta = np.random.uniform(0, 2 * np.pi)
                        self.robot.reset_robot(self.body, x, y, theta)

                    elif self.mode == "easy-01":
                        self.new_map_path, self.segments, self.poly = self.generator.world(
                            self.grid_length
                        )
                        targets = np.array([[0.35, 0.35], [0.35, 1.8], [1.8, 0.35], [1.8, 1.8]])
                        choice = targets[np.random.randint(0, len(targets))]
                        self.target_x, self.target_y = choice[0], choice[1]
                        x, y = 1.07, 1.07

                        theta = np.random.uniform(0, 2 * np.pi)
                        self.robot.reset_robot(self.body, x, y, theta)

                    elif self.mode in ("easy-02", "easy-03", "easy-04"):
                        self.new_map_path, self.segments, self.poly = self.generator.world(
                            self.grid_length
                        )
                        robot_pos, goal_pos = spawn_robot_and_goal(
                            poly=self.poly,
                            robot_clearance=self.threshold,
                            goal_clearance=self.collision,
                            min_robot_goal_dist=0.03,
                        )
                        self.target_x, self.target_y = goal_pos[0], goal_pos[1]
                        x, y = robot_pos[0], robot_pos[1]

                        self.sensor.update_map(self.segments)

                        theta = np.random.uniform(0, 2 * np.pi)
                        self.robot.reset_robot(self.body, x, y, theta)

                    elif self.mode in ("easy-05"):
                        self.grid_length = round(np.random.choice(np.arange(2, 10.05, 0.05)), 2)
                        print(self.grid_length)

                        self.new_map_path, self.segments, self.poly = self.generator.world(
                            self.grid_length
                        )
                        robot_pos, goal_pos = spawn_robot_and_goal(
                            poly=self.poly,
                            robot_clearance=self.threshold,
                            goal_clearance=self.collision,
                            min_robot_goal_dist=0.03,
                        )
                        self.target_x, self.target_y = goal_pos[0], goal_pos[1]
                        x, y = robot_pos[0], robot_pos[1]

                        self.sensor.update_map(self.segments)

                        theta = np.random.uniform(0, 2 * np.pi)
                        self.robot.reset_robot(self.body, x, y, theta)

                    elif "medium" in self.mode:
                        self.new_map_path, self.segments, self.poly = self.create_world.world(
                            mode=self.mode
                        )
                        self.sensor.update_map(self.segments)
                        robot_pos, goal_pos = spawn_robot_and_goal(
                            poly=self.poly,
                            robot_clearance=self.threshold,
                            goal_clearance=self.collision,
                            min_robot_goal_dist=0.03,
                        )
                        self.target_x, self.target_y = goal_pos[0], goal_pos[1]
                        x, y = robot_pos[0], robot_pos[1]

                        theta = np.random.uniform(0, 2 * np.pi)
                        self.robot.reset_robot(self.body, x, y, theta)

                    elif "hard" in self.mode:
                        self.new_map_path, self.segments, self.poly = self.generator.world(
                            self.grid_length
                        )
                        self.sensor.update_map(self.segments)
                        robot_pos, goal_pos = spawn_robot_and_goal(
                            poly=self.poly,
                            robot_clearance=self.threshold,
                            goal_clearance=self.collision,
                            min_robot_goal_dist=0.03,
                        )
                        self.target_x, self.target_y = goal_pos[0], goal_pos[1]
                        x, y = robot_pos[0], robot_pos[1]

                        theta = np.random.uniform(0, 2 * np.pi)
                        self.robot.reset_robot(self.body, x, y, theta)

                    intersections, measurement = self.sensor.sensor(
                        x=self.body.position.x,
                        y=self.body.position.y,
                        theta=self.body.position.angle,
                        max_range=self.max_lidar,
                    )

                    dist = distance_to_goal(
                        self.body.position.x,
                        self.body.position.y,
                        self.target_x,
                        self.target_y,
                        self.max_dist,
                    )
                    alpha = angle_to_goal(
                        self.body.position.x,
                        self.body.position.y,
                        self.body.position.angle,
                        self.target_x,
                        self.target_y,
                        self.max_alpha
                    )

                    self.initial_distance = dist

                    self.current_rays = len(measurement)
                    padded_lidar = np.zeros((self.max_num_rays,), dtype=np.float32)
                    padded_lidar[: self.current_rays] = measurement[: self.current_rays]

                    lidar_norm = self.scaler_lidar.transform(
                        padded_lidar.reshape(1, -1)
                    ).flatten()
                    dist_norm = self.scaler_dist.transform(
                        np.array(dist).reshape(1, -1)
                    ).flatten()
                    alpha_norm = self.scaler_alpha.transform(
                        np.array(alpha).reshape(1, -1)
                    ).flatten()

                    action = np.random.randint(0, 3)
                    action_one_hot = np.eye(3)[action]
                    min_lidar_norm = np.min(lidar_norm)

                    states = np.concatenate(
                        (
                            np.array(lidar_norm, dtype=np.float32),
                            np.array(action_one_hot, dtype=np.int16),
                            np.array(dist_norm, dtype=np.float32),
                            np.array(alpha_norm, dtype=np.float32),
                        )
                    )
                    self.last_states = states

                    if self.use_render:
                        for patch in self.ax.patches:
                            patch.remove()
                        self.ax.add_patch(self.new_map_path)
                        art3d.pathpatch_2d_to_3d(self.new_map_path, z=0, zdir="z")

                        self._plot_anim(
                            0,
                            intersections,
                            self.body.position.x,
                            self.body.position.y,
                            self.target_x,
                            self.target_y,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            alpha_norm,
                            min_lidar_norm,
                            dist_norm,
                            self.action,
                        )

                except Exception as e:
                    print(
                        f"[RESET-ERROR] Erro ao configurar o cenário (mode = {self.mode}): {e}"
                    )
                    raise

                info = {}
                return states, info
        """

        prompt = (
            "Você é um assistente para configurar o treinamento RL de robôs.\n\n"
            f"Historico das ultimas avaliacoes: {context_info}\n\n"
            # f"Codigo do ambiente: {code_str}\n\n"
            "**1. Configurações Básicas:**\n"
            f"   - Base de configurações: {base_info}\n"
            f"   - Métricas de treinamento: {stats_info}\n\n"
            f"   - F(pi) consiste na porcentagem de acerto de o agente chegar ao objetivo em 10 epsodios: {stats_info}\n\n"
            "**2. Contexto sobre as estados / recompensas / ambiente: **\n"
            "   - Todos os estados varia de 0 a 1. Ou seja todos sao nromalizados para esses valores.\n"
            "   - A recompensa varia de -1 a 1 independente do scale selecionado\n"
            "   - O Alpha: varia de 0 a 1. 0 significa que o robo esta em direcao ao objetivo e 1 esta ao contrario do objetivo,\n"
            "   - O Dist: varia de 0 a 1. Quando mais proximo de 0 mais perto o robo esta do objetivo.\n"
            "   - O timestep maximo do ambiente é 1000."
            "**3. Detalhes das Métricas (Desvio padrão e média):**\n"
            "   - obstacle_score: penalidade quando o sensor lidar tem medicoes muito perto de obstaculos\n"
            "   - orientation_score: maior recompensa se o robô estiver direcionado para o objetivo.\n"
            "   - progress_score: diferença entre a posição inicial e a posição atual do robô em relação ao objetivo.\n"
            "   - time_score: penalidade por tempo.\n"
            "   - min_lidar: menor medição do lidar.\n"
            "**4. Tarefa:**\n"
            "   Usando as métricas e a base de configurações, avalie e retorne em formato JSON:\n"
            "     - Precisa avaliar se o agente esta conseguindo chegar ao objetivo com base na metrica de F(pi), leve em consideracao as metricas e recompensas\n"
            "     - Escolha o o tamanho do mapa de 1 a 5 de tamanho, se for 1 é mais facil e assim por diante até chegar no 5 que é maior\n\n"
            "     - Escolha a porcentagem de obstaculos que varia de 0% a 50%. 0% é sem obstaculos e 50% é com muitos obstaculos\n\n"
            "     - Escolha a escala da recompensa e se vai ser positivo ou nao\n"
            "**Observações**\n"
            "* O objetivo é ensinar o robô a chegar ao alvo sem colidir com obstáculos, usando apenas os estados (medições do lidar, ângulo alpha, "
            "distância até o objetivo e a última ação tomada).\n"
            "* O robô deve ser capaz de aprender a melhor política de ação para maximizar a recompensa total.\n"
            "* Justifique suas escolhas no parametro 'justify'.\n"
            "* Avalie se o agente esta pronto para o proximo nivel de dificuldade do ambiente.\n"
            "* Sempre responda em portugues nas avaliacoes\n"
        )

        return prompt

    def call_evaluator_llm(self, prompt: str):
        """
        Calls the Gemini LLM using the google.generativeai package.
        """
        client = genai.Client(api_key=self.evaluator_api_key)

        manual_schema = {
            "type": "object",
            "properties": {
                "justify": {"type": "string"},
                "strategy": {
                    "type": "object",
                    "properties": {
                        "reward": {
                            "type": "object",
                            "properties": {
                                "reward_type": {"type": "string"},
                                "parameters": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "key": {"type": "string"},
                                            "value": {"type": "number"},
                                        },
                                        "required": ["key", "value"],
                                    },
                                },
                            },
                            "required": ["reward_type", "parameters"],
                        },
                        "domain": {
                            "type": "object",
                            "properties": {
                                "obstacle_percentage": {
                                    "type": "object",
                                    "properties": {
                                        "value": {"type": "integer"},
                                        "description": {"type": "string"},
                                    },
                                    "required": ["value", "description"],
                                },
                                "map_size": {
                                    "type": "object",
                                    "properties": {
                                        "value": {"type": "number"},
                                        "description": {"type": "string"},
                                    },
                                    "required": ["value", "description"],
                                },
                            },
                            "required": ["obstacle_percentage", "map_size"],
                        },
                    },
                    "required": ["reward", "domain"],
                },
            },
            "required": ["strategy"],
        }

        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=manual_schema,
                temperature=0,
                top_p=0.95,
                top_k=20,
                candidate_count=1,
                seed=5,
            ),
        )

        if response is not None:
            return response.parsed
        else:
            raise ValueError("No candidates returned from Gemini API.")

    def evaluate_training(self, context: dict, stats: dict, history: dict):
        """
        Generates a prompt, calls the LLM, and returns the evaluation text.
        """
        prompt = self.build_evaluation_prompt(context, stats, history)
        llm_response = self.call_evaluator_llm(prompt)
        return llm_response
