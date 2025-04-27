import time

import gymnasium as gym
import psutil


def check_system_usage():
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent()
    return mem.used / (1024**3), cpu


def test_env_capacity(env_constructor, max_envs=100, mem_limit_gb=10.0, cpu_limit=80.0):
    """
    Tenta criar progressivamente ambientes até estourar os limites.
    env_constructor: função que retorna um novo ambiente (ex.: lambda: gym.make('CartPole-v1'))
    max_envs: limite máximo de tentativas
    mem_limit_gb: uso máximo de RAM (GB)
    cpu_limit: uso máximo de CPU (%)
    """
    envs = []
    for i in range(1, max_envs + 1):
        envs.append(env_constructor())  # cria mais um ambiente
        time.sleep(1)  # espera pra estabilizar

        mem_used_gb, cpu_used = check_system_usage()
        print(
            f"Ambientes: {i}, Memória usada: {mem_used_gb:.2f} GB, CPU usada: {cpu_used:.1f}%"
        )

        if mem_used_gb > mem_limit_gb or cpu_used > cpu_limit:
            print("Limite atingido. Liberando ambientes...")
            # Fecha ambientes, se necessário
            for e in envs:
                e.close()
            return i - 1  # O anterior foi o máximo seguro

    # Se chegamos até aqui, testamos tudo e não estourou
    for e in envs:
        e.close()
    return max_envs


if __name__ == "__main__":
    # Exemplo de uso: testando criar ambientes CartPole
    max_safe_envs = test_env_capacity(lambda: gym.make("CartPole-v1"), max_envs=20)
    print(f"Você pode rodar ~{max_safe_envs} ambientes de forma segura.")
