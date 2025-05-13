#!/usr/bin/env python3
import math, os, random

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import LaserScan
import torch
import torch.nn as nn
from torch.distributions import Categorical
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient

MAX_DISTANCE = 2.6593232221751464
MIN_DISTANCE = 0.0
MAX_LIDAR_RANGE = 5.0
MIN_LIDAR_RANGE = 0.0
MAX_ALPHA = 3.5 * 0.89
MIN_ALPHA = 0.0
VEL_LINEAR = 0.22
VEL_ANGULAR = 2.84

def distance_to_goal(
    x: float, y: float, goal_x: float, goal_y: float, max_value: float
) -> float:
    dist = np.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)
    if dist >= max_value:
        return max_value
    else:
        return dist


def angle_to_goal(x, y, theta, goal_x, goal_y, max_value):
    ox, oy = math.cos(theta), math.sin(theta)          # orientação
    gx, gy = goal_x - x, goal_y - y                    # vetor p/ alvo

    # cross escalar 2-D
    cross_val = ox * gy - oy * gx                      # -> float32 já ok
    dot_val   = ox * gx + oy * gy

    alpha = abs(math.atan2(abs(cross_val), dot_val))
    return max_value if alpha >= max_value else alpha

def block(in_f, out_f):
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.LayerNorm(out_f),
        nn.LeakyReLU(),
    )


class PolicyBackbone(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.body = nn.Sequential(
            block(feature_dim, 32),
            block(32, 32),
            block(32, 16),
        )


class RNLPolicy(nn.Module):
    def __init__(self, in_dim: int, n_act: int,
                 pth: str, device: str = "cpu"):
        super().__init__()
        self.backbone = PolicyBackbone(in_dim)
        self.head = nn.Linear(16, n_act)

        ckpt = torch.load(pth, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        ckpt = _map_sb3_keys(ckpt)           # renomeia
        missing, unexpected = self.load_state_dict(ckpt, strict=False)
        assert not missing, f"Pesos faltando: {missing}"
        self.eval()

    @torch.no_grad()
    def act(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        logits = self.head(self.backbone.body(obs))
        return Categorical(logits=logits).sample().item()


def _map_sb3_keys(sd: dict) -> dict:
    new = {}
    for k, v in sd.items():
        if k.startswith("mlp_extractor.policy_net."):
            new_k = k.replace("mlp_extractor.policy_net.", "backbone.body.")
        elif k.startswith("action_net."):
            new_k = k.replace("action_net.", "head.")
        else:
            continue
        new[new_k] = v
    return new

def quat_to_yaw(x, y, z, w):
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)


class CustomMinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.data_min = None
        self.data_max = None
        self.scale_ = None
        self.min_ = None

    def fit(self, X):
        X = np.array(X)
        self.data_min = np.min(X, axis=0)
        self.data_max = np.max(X, axis=0)
        data_range = self.data_max - self.data_min
        data_range[data_range == 0] = 1
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / data_range
        self.min_ = self.feature_range[0] - self.data_min * self.scale_
        return self

    def transform(self, X):
        X = np.array(X)
        return X * self.scale_ + self.min_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

class InferenceModel(Node):
    def __init__(self):
        super().__init__('sim_environment')
        self.position = None
        self.last_states = np.zeros(8)
        self.action = 0
        self.target = False

        self.goal_positions = [(0.218, -1.247), (0.063, -0.568), (0.218, -1.247), (0.932, -1.259), (0.218, -1.247), (0.226, -2.0677), (0.218, -1.247), (-0.438, -1.386)]
        self.goal_order = random.sample(range(len(self.goal_positions)),
                                        len(self.goal_positions))
        self.goal_index = 0
        self.goal_x, self.goal_y = self.goal_positions[self.goal_order[0]]

        self.max_num_rays = 5
        self.goto_x = -0.00859
        self.goto_y = -1.28529
        self.lidar_ranges = [0.0] * self.max_num_rays
        self.i_update = 0
        self.update_rates = 2

        pkg_dir = get_package_share_directory('playground')
        model_path = os.path.join(pkg_dir, 'models', 'policy.pth')
        self.policy = RNLPolicy(in_dim=8, n_act=3, pth=model_path)

        self.scaler_lidar = CustomMinMaxScaler(feature_range=(0, 1))
        self.scaler_dist = CustomMinMaxScaler(feature_range=(0, 1))
        self.scaler_alpha = CustomMinMaxScaler(feature_range=(0, 1))

        self.max_lidar, self.min_lidar = MAX_LIDAR_RANGE, MIN_LIDAR_RANGE
        self.scaler_lidar.fit(
            np.array(
                [
                    [self.min_lidar] * self.max_num_rays,
                    [self.max_lidar] * self.max_num_rays,
                ]
            )
        )
        self.scaler_dist.fit(np.array([[MIN_DISTANCE], [MAX_DISTANCE]]))
        self.scaler_alpha.fit(np.array([[MIN_ALPHA], [MAX_ALPHA]]))

        qos = QoSProfile(depth=10,
                         reliability=QoSReliabilityPolicy.RELIABLE,
                         durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        lidar_qos = QoSProfile(depth=5,
                               reliability=QoSReliabilityPolicy.BEST_EFFORT,
                               durability=QoSDurabilityPolicy.VOLATILE)

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.scan_sub = self.create_subscription(LaserScan, '/scan',
                                                 self.laser_callback, lidar_qos)
        self.amcl_sub = self.create_subscription(PoseWithCovarianceStamped,
                                                 '/amcl_pose',
                                                 self.amcl_callback, qos)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.update)

    # ---------- callbacks ----------
    def laser_callback(self, msg):
        step = (len(msg.ranges) - 1) / 4.0
        for i in range(5):
            idx = min(int(round(step * i)), len(msg.ranges)-1)
            val = msg.ranges[idx] if math.isfinite(msg.ranges[idx]) else msg.range_max
            self.lidar_ranges[i] = val

    def amcl_callback(self, msg):
        self.position = msg.pose.pose

    # ---------- robo ----------
    def move_robot(self, action):
        if action == 0:
            vl = VEL_LINEAR/2
            vr = 0.0
        elif action == 1:
            vl = VEL_LINEAR/6
            vr = -VEL_ANGULAR/2
        elif action == 2:
            vl = VEL_LINEAR/6
            vr = VEL_ANGULAR/2
        else:
            vl, vr = 0.0, 0.0

        twist = Twist()
        twist.linear.x = vl
        twist.angular.z = vr
        self.cmd_vel_pub.publish(twist)

    def stop_robot(self):
        self.cmd_vel_pub.publish(Twist())

    # ---------- loop ----------
    def update(self):
        if self.position is None:
            self.get_logger().debug('Aguardando /amcl_pose…')
            return

        if any(r < 0.3 for r in self.lidar_ranges):
            self.get_logger().warn('Obstáculo! Parando robô...')
            self.stop_robot()
            self.action = 3
            return

        else:
            self.action = self.policy.act(self.last_states)
            self.move_robot(self.action)

        x, y = self.position.position.x, self.position.position.y
        q = self.position.orientation
        theta = quat_to_yaw(q.x, q.y, q.z, q.w)

        dist = distance_to_goal(x, y, self.goal_x, self.goal_y, MAX_DISTANCE)
        alpha = angle_to_goal(x, y, theta, self.goal_x, self.goal_y, MAX_ALPHA)

        lidar_array = np.array(self.lidar_ranges, dtype=np.float32)
        lidar_norm = self.scaler_lidar.transform(lidar_array.reshape(1, -1)).flatten()
        dist_norm = self.scaler_dist.transform(np.array(dist).reshape(1, -1)).flatten()
        alpha_norm = self.scaler_alpha.transform(
            np.array(alpha).reshape(1, -1)
        ).flatten()

        action_one_hot = [np.int16(self.action != 0)]
        self.last_states = np.concatenate(
            (
                np.array(lidar_norm, dtype=np.float32),
                np.array(action_one_hot, dtype=np.int16),
                np.array(dist_norm, dtype=np.float32),
                np.array(alpha_norm, dtype=np.float32),
            )
        )

        self.i_update += 1

        # self.get_logger().warn(f'States: {self.last_states}')

        if dist <= 0.2:
            self.goal_index = (self.goal_index + 1) % len(self.goal_positions)
            self.goal_x, self.goal_y = self.goal_positions[self.goal_order[self.goal_index]]
            self.get_logger().info(f'Nova meta: ({self.goal_x:.2f}, {self.goal_y:.2f})')

def main(args=None):
    rclpy.init(args=args)
    node = InferenceModel()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Ctrl-C detectado, parando robô…')
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
