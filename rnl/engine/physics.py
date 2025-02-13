import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

ti.init(arch=ti.cpu)

# Parâmetros
N_ANGLES = 360            # Número de raios do lidar
N_SEGMENTS = 4            # Segmentos do quadrado (mundo)
lidar_range = 5.0         # Alcance do lidar
dt = 0.1                  # Passo de tempo

# Campos Taichi
lidar_angles = ti.field(dtype=ti.f32, shape=N_ANGLES)
segments = ti.Vector.field(4, dtype=ti.f32, shape=N_SEGMENTS)
points = ti.Vector.field(2, dtype=ti.f32, shape=N_ANGLES)
distances = ti.field(dtype=ti.f32, shape=N_ANGLES)

@ti.kernel
def compute_lidar(robot_x: ti.f32, robot_y: ti.f32, robot_theta: ti.f32, lidar_range: ti.f32):
    for i in range(N_ANGLES):
        ang = lidar_angles[i] + robot_theta
        ray = ti.Vector([lidar_range * ti.cos(ang), lidar_range * ti.sin(ang)])
        best_dist = lidar_range
        best_pt = ti.Vector([0.0, 0.0])
        p = ti.Vector([robot_x, robot_y])
        for j in range(N_SEGMENTS):
            seg = segments[j]
            q = ti.Vector([seg[0], seg[1]])
            s = ti.Vector([seg[2] - seg[0], seg[3] - seg[1]])
            rxs = ray[0] * s[1] - ray[1] * s[0]
            if ti.abs(rxs) < 1e-6:
                continue
            qp = q - p
            t = (qp[0] * s[1] - qp[1] * s[0]) / rxs
            u = (qp[0] * ray[1] - qp[1] * ray[0]) / rxs
            if 0 <= t <= 1 and 0 <= u <= 1:
                ip = p + t * ray
                d = (ip - p).norm()
                if d < best_dist:
                    best_dist = d
                    best_pt = ip
        points[i] = best_pt
        distances[i] = best_dist

# Inicializa os ângulos do lidar (0 a 2pi)
angles = np.linspace(0, 2*np.pi, N_ANGLES, endpoint=False).astype(np.float32)
for i in range(N_ANGLES):
    lidar_angles[i] = angles[i]

# Define os segmentos do mundo (quadrado de 0 a 5)
world_segments = np.array([
    [0.0, 0.0, 5.0, 0.0],
    [5.0, 0.0, 5.0, 5.0],
    [5.0, 5.0, 0.0, 5.0],
    [0.0, 5.0, 0.0, 0.0],
], dtype=np.float32)
for i in range(N_SEGMENTS):
    segments[i] = ti.Vector([world_segments[i,0], world_segments[i,1],
                              world_segments[i,2], world_segments[i,3]])

# Estado inicial do robô (posição e orientação)
robot_x = 2.5
robot_y = 2.5
robot_theta = 0.0

# Guarda a trajetória para visualização
robot_path = [(robot_x, robot_y)]

# Controlador simples:
# Se o lidar na frente (índice 0) detectar obstáculo muito perto, roda;
# Caso contrário, segue em frente.
def controller(front_distance, threshold=0.5):
    if front_distance < threshold:
        return 0.0, 1.0   # Para, gira
    else:
        return 0.2, 0.0   # Vai pra frente

# Configura o plot 3D com Matplotlib
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1, 6)
ax.set_ylim(-1, 6)
ax.set_zlim(-1, 6)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Simulação 3D de Robô Diferencial com Lidar (Mundo 2D)")

# Marcadores e linhas em 3D (z=0)
robot_marker, = ax.plot([], [], [], 'bo', markersize=8)
orientation_line, = ax.plot([], [], [], 'b-', lw=2)
num_lines = 36
lidar_lines = []
for _ in range(num_lines):
    line, = ax.plot([], [], [], 'r-', lw=0.5, alpha=0.5)
    lidar_lines.append(line)
path_line, = ax.plot([], [], [], 'g--', lw=1)

# Desenha o mundo (quadrado) em z=0
for seg in world_segments:
    ax.plot([seg[0], seg[2]], [seg[1], seg[3]], [0, 0], 'k-', lw=2)

# Inicializa o plot
def init():
    robot_marker.set_data([], [])
    robot_marker.set_3d_properties([])
    orientation_line.set_data([], [])
    orientation_line.set_3d_properties([])
    path_line.set_data([], [])
    path_line.set_3d_properties([])
    for line in lidar_lines:
        line.set_data([], [])
        line.set_3d_properties([])

init()
plt.ion()  # ativa o modo interativo

# Loop de renderização
for frame in range(300):
    # Atualiza as medições do lidar e a cinemática do robô
    compute_lidar(robot_x, robot_y, robot_theta, lidar_range)
    pts = points.to_numpy()
    dists = distances.to_numpy()

    # Controle usando o raio frontal (índice 0)
    front_distance = dists[0]
    v, omega = controller(front_distance)

    # Atualiza a cinemática do robô (modelo diferencial)
    robot_x += v * np.cos(robot_theta) * dt
    robot_y += v * np.sin(robot_theta) * dt
    robot_theta += omega * dt
    robot_path.append((robot_x, robot_y))

    # Atualiza o marcador do robô (em 3D, z=0)
    robot_marker.set_data([robot_x], [robot_y])
    robot_marker.set_3d_properties([0])

    # Atualiza a linha de orientação do robô
    line_length = 0.5
    x_end = robot_x + line_length * np.cos(robot_theta)
    y_end = robot_y + line_length * np.sin(robot_theta)
    orientation_line.set_data([robot_x, x_end], [robot_y, y_end])
    orientation_line.set_3d_properties([0, 0])

    # Atualiza os raios do lidar (apenas alguns para visualização)
    indices = np.linspace(0, N_ANGLES-1, len(lidar_lines), dtype=np.int32)
    for idx, line in zip(indices, lidar_lines):
        pt = pts[idx]
        line.set_data([robot_x, pt[0]], [robot_y, pt[1]])
        line.set_3d_properties([0, 0])

    # Atualiza a trajetória do robô
    path_np = np.array(robot_path)
    path_line.set_data(path_np[:,0], path_np[:,1])
    path_line.set_3d_properties(np.zeros(path_np.shape[0]))

    plt.draw()
    plt.pause(0.05)  # pausa de 50ms para atualização

plt.ioff()
plt.show()
