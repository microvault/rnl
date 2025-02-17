from numba import njit
import numpy as np
import matplotlib.pyplot as plt

numero_feixes = 360
maxima_alcance = 10.0
numero_de_segmentos = 4

types = np.float32

# Definindo os obstáculos como segmentos (cada segmento tem 2 pontos)
obstaculos = np.array(
    [
        [[0.0, 0.0], [2.0, 0.0]],
        [[2.0, 0.0], [2.0, 2.0]],
        [[2.0, 2.0], [0.0, 2.0]],
        [[0.0, 2.0], [0.0, 0.0]],
    ],
    dtype=types,
)

# Posição do lidar em x, y
posicao_lidar = np.array([1.0, 1.0], dtype=types)
# ponto de impacto para cada feixe
resultados = np.zeros((numero_feixes, 2), dtype=types)

@njit(fastmath=True)
def produto_vetorial_2d(a, b) -> float:
    # produto vetorial entre dois vetores 2D, retorna um escalar
    return a[0] * b[1] - a[1] * b[0]

def interseccao_raio_lidar(origem, direcao, p1, p2):
    # inicia o valor de t em -1 para indicar sem intersecao
    reta_t = -1.0
    # so sera usando quando tiver interseccao
    reta_ponto = np.zeros(2, dtype=types)

    # v1 é o ponto que vai da origem (ponto inicial) ate o ponto p1
    v1 = origem - p1
    # v2 é o vetor que representa o segmento de p1 a p2
    v2 = p2 - p1
    # v3 é o vetor perpendicular a direcao do raio
    v3 = np.empty(2, dtype=types)
    v3[0] = -direcao[1]
    v3[1] = direcao[0]

    # calcula o produto escalar entre v2 e v3 para verificar se o raio e o segmento sao paralelos
    valor_do_denominador = v2[0] * v3[0] + v2[1] * v3[1]

    if np.abs(valor_do_denominador) >= 1e-6:
        # calcula o valor de t que representa a distancia ao longo do raio
        t = produto_vetorial_2d(v2, v1) / valor_do_denominador
        # u representa a posicao ao longo do segmento
        u = (v1[0] * v3[0] + v1[1] * v3[1]) / valor_do_denominador
        # t deve ser maior que 0 que significa que esta a frente da origem
        # u precisa estar entre 0 e 1 (a interseccao esta dentro do segmento)
        if t >= 0.0 and u >= 0.0 and u <= 1.0:
            reta_t = t
            # calcula o ponto de interseccao
            reta_ponto = origem + t * direcao

    # retorna o valor t (distancia ao longo do raio) e o ponto de intersecacao
    return reta_t, reta_ponto


# Define a origem do raio (por exemplo, posição do LiDAR)
origem = np.array([2.5, 2.5], dtype=np.float32)
# Define a direção do raio (por exemplo, para a direita)
direcao = np.array([1.0, 0.0], dtype=np.float32)
# Define um segmento vertical que pode interceptar o raio
p1 = np.array([5.0, 1.0], dtype=np.float32)
p2 = np.array([5.0, 4.0], dtype=np.float32)

# Chama a função de interseção
t, ponto = interseccao_raio_lidar(origem, direcao, p1, p2)

print("t =", t)
print("Ponto de interseção =", ponto)


# --- Plot para Visualizar ---

plt.figure(figsize=(8,8))

# Plota o segmento como uma linha preta
plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=2, label="Segmento")

# Se houver interseção (t >= 0), plota o raio até o ponto de interseção; caso contrário, plota um raio de comprimento arbitrário.
if t >= 0:
    ray_end = ponto
else:
    ray_end = origem + 10.0 * direcao  # comprimento arbitrário

# Plota o raio (linha vermelha)
plt.plot([origem[0], ray_end[0]], [origem[1], ray_end[1]], 'r-', linewidth=2, label="Raio")

# Plota a origem (ponto azul)
plt.plot(origem[0], origem[1], 'bo', markersize=8, label="Origem")

# Se houver interseção, plota o ponto de interseção (ponto verde)
if t >= 0:
    plt.plot(ponto[0], ponto[1], 'go', markersize=8, label="Interseção")

plt.xlim(0, 6)
plt.ylim(0, 6)
plt.legend()
plt.grid(True)
plt.show()
