#!/usr/bin/env python3
import math, os, random, signal, sys
from pathlib import Path

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


def clamp(value, vmin, vmax):
    return max(vmin, min(value, vmax))


def distance_to_goal(x, y, goal_x, goal_y):
    return min(np.hypot(x - goal_x, y - goal_y), 9.0)


def angle_to_goal(x, y, theta, goal_x, goal_y):
    o_t = np.array([np.cos(theta), np.sin(theta)])
    g_t = np.array([goal_x - x, goal_y - y])
    return float(abs(np.arctan2(np.cross(o_t, g_t), np.dot(o_t, g_t))))


class FlexMLP(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


def _map_sb3_keys(sd):
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


class RNLPolicy(nn.Module):
    def __init__(self, in_dim, n_act, hidden, pth, device="cpu"):
        super().__init__()
        self.backbone = FlexMLP(in_dim, hidden)
        self.head = nn.Linear(hidden[-1], n_act)

        ckpt = torch.load(pth, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        ckpt = _map_sb3_keys(ckpt)
        self.load_state_dict(ckpt, strict=False)
        self.eval()

    @torch.no_grad()
    def act(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return Categorical(logits=self.head(self.backbone(obs))).sample().item()


class InferenceModel(Node):
    def __init__(self):
        super().__init__('sim_environment')
        self.position = None
        self.last_states = np.zeros(8)

        self.goal_positions = [(3.0925, 2.6864), (4.6748, 1.3596),
                               (3.7367, -0.2997), (2.0904, 0.6386)]
        self.goal_order = random.sample(range(len(self.goal_positions)),
                                        len(self.goal_positions))
        self.goal_index = 0
        self.goal_x, self.goal_y = self.goal_positions[self.goal_order[0]]

        self.lidar_ranges = [0.0] * 5
        self.data_min = [0.5]*5 + [1.0, 0.0, 0, 0, 0]
        self.data_max = [3.5]*5 + [9.0, 3.5, 1, 1, 1]

        pkg_dir = get_package_share_directory('playground')
        model_path = os.path.join(pkg_dir, 'models', 'policy.pth')
        self.policy = RNLPolicy(8, 3, [20, 64], model_path)

        qos = QoSProfile(depth=10,
                         reliability=QoSReliabilityPolicy.RELIABLE,
                         durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        lidar_qos = QoSProfile(depth=5,
                               reliability=QoSReliabilityPolicy.BEST_EFFORT,
                               durability=QoSDurabilityPolicy.VOLATILE)

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
            vl, vr = 0.129, 0.0
        elif action == 1:
            vl, vr = 0.08, 0.7
        elif action == 2:
            vl, vr = 0.08, -0.7
        else:
            vl, vr = 0.0, 0.0

        twist = Twist()
        twist.linear.x = vl
        twist.angular.z = vr
        self.cmd_vel_pub.publish(twist)

    def stop_robot(self):
        self.cmd_vel_pub.publish(Twist())  # zero tudo

    # ---------- loop ----------
    def update(self):
        if self.position is None:
            self.get_logger().debug('Aguardando /amcl_pose…')
            return

        action = self.policy.act(self.last_states)
        self.move_robot(action)
        self.get_logger().info(f'Meta: ({self.goal_x:.2f}, {self.goal_y:.2f})')

        x, y = self.position.position.x, self.position.position.y
        z, w = self.position.orientation.z, self.position.orientation.w
        theta = 2.0 * math.atan2(z, w)

        dist = distance_to_goal(x, y, self.goal_x, self.goal_y)
        alpha = angle_to_goal(x, y, theta, self.goal_x, self.goal_y)

        norm_lidar = (np.clip(self.lidar_ranges, 0.5, 3.5) - 0.5) / 3.0
        action_one_hot = [int(action != 0)]
        self.last_states = np.concatenate([norm_lidar,
                                           action_one_hot,
                                           [dist/9.0, alpha/3.5]]).astype(np.float32)

        if dist <= 0.8:
            self.goal_index = (self.goal_index + 1) % len(self.goal_positions)
            self.goal_x, self.goal_y = self.goal_positions[self.goal_order[self.goal_index]]
            self.get_logger().info(f'Nova meta: ({self.goal_x:.2f}, {self.goal_y:.2f})')


# -------------------- main --------------------
def main(args=None):
    rclpy.init(args=args)
    node = InferenceModel()

    try:
        rclpy.spin(node)               # fica rodando até Ctrl-C
    except KeyboardInterrupt:
        node.get_logger().info('Ctrl-C detectado, parando robô…')
    finally:
        node.stop_robot()              # zera velocidade
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
