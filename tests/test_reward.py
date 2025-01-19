import csv

import matplotlib.pyplot as plt
import numpy as np


def test_plot_reward_separated(csv_file: str = "rewards.csv", num_elements: int = None):
    collision_scores = []
    orientation_scores = []
    progress_scores = []
    time_scores = []
    rewards = []

    # Ler o arquivo CSV
    try:
        with open(csv_file, mode="r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                collision_scores.append(float(row["collision_score"]))
                orientation_scores.append(float(row["orientation_score"]))
                progress_scores.append(float(row["progress_score"]))
                time_scores.append(float(row["time_score"]))
                rewards.append(float(row["reward"]))
                if num_elements is not None and len(collision_scores) >= num_elements:
                    break
    except FileNotFoundError:
        print(f"O arquivo {csv_file} não foi encontrado.")
        return
    except Exception as e:
        print(f"Ocorreu um erro ao ler o arquivo CSV: {e}")
        return

    if not collision_scores:
        print("Nenhum dado encontrado no arquivo CSV.")
        return

    # Gerar o eixo X como o índice das linhas
    steps = list(range(1, len(rewards) + 1))

    # Definir uma lista de componentes para facilitar a iteração
    components = [
        ("Collision Score", collision_scores, "red"),
        ("Orientation Score", orientation_scores, "green"),
        ("Progress Score", progress_scores, "blue"),
        ("Time Score", time_scores, "orange"),
        ("Total Reward", rewards, "purple"),
    ]

    # Configurar o layout da figura com subplots
    num_plots = len(components)
    cols = 2
    rows = (num_plots + cols - 1) // cols  # Calcula o número de linhas necessárias

    plt.figure(figsize=(10, 5 * rows))  # Ajusta a altura com base no número de linhas

    for idx, (title, data, color) in enumerate(components, 1):
        ax = plt.subplot(rows, cols, idx)
        ax.plot(steps, data, label=title, color=color, linestyle="-", linewidth=1.5)
        ax.set_ylabel(title, fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)

        # Calcular estatísticas
        mean_val = np.mean(data)
        min_val = np.min(data)
        max_val = np.max(data)

        # Adicionar texto abaixo do plot com as estatísticas
        # A posição (0.5, -0.25) coloca o texto centralizado abaixo do gráfico
        ax.text(
            0.5,
            -0.25,
            f"Média: {mean_val:.2f} | Mínimo: {min_val:.2f} | Máximo: {max_val:.2f}",
            transform=ax.transAxes,
            ha="center",
            fontsize=12,
        )

    # Ajustar layout para evitar sobreposição
    plt.tight_layout()

    # Ajustar espaço adicional na parte inferior para os textos
    plt.subplots_adjust(bottom=0.15)

    # Exibir o gráfico
    plt.show()


# Chamada da função com o caminho do arquivo CSV
test_plot_reward_separated(csv_file="/Users/nicolasalan/microvault/rnl/debugging.csv")
