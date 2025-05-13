import numpy as np
import matplotlib.pyplot as plt

metrics = [
    'Sucesso',
    'Zona Insegura',
    'Instabilidade',
    'Tempo até Colisão',
    'Tempo até Objetivo',
]

# valores já em porcentagem
com_pct = [100.0, 0.00, 17.92, 0.00, 26.62]
sem_pct = [ 20.0, 0.10, 10.10, 2.12,  6.24]

x  = np.arange(len(metrics))
bw = 0.35

fig, ax = plt.subplots(figsize=(9, 4))

bars1 = ax.bar(x - bw/2, com_pct, bw,
               label='LLM', color='#FFE975', hatch='//',
               edgecolor='black', linewidth=0.8)
bars2 = ax.bar(x + bw/2, sem_pct, bw,
               label='Humano', color='#4C72B0',
               edgecolor='black', linewidth=0.8)

# anotar porcentagens
for bars, vals in [(bars1, com_pct), (bars2, sem_pct)]:
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 2,
                f'{val:.2f}%', ha='center',
                va='bottom', fontsize=8, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=9)
ax.set_ylabel('Valor (%)')
ax.set_ylim(0, 100)
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.legend(title='Fonte do feedback', frameon=False, loc='upper right')

plt.tight_layout()
plt.savefig('result_long.png', dpi=300)
