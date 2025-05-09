import numpy as np
import matplotlib.pyplot as plt

metrics = [
    'Sucesso (%)',
    'Zona Insegura (%)',
    'Instabilidade (%)',
    'Tempo até Colisão',
    'Tempo até Objetivo',
]
com_raw = [90.0, 0.00, 2.34,  0.00, 39.11]
sem_raw = [90.0, 0.02, 3.95, 41.00, 67.67]

ranges_max = [100, 1, 10, 100, 100]
com_norm   = np.array(com_raw) / ranges_max * 100
sem_norm   = np.array(sem_raw) / ranges_max * 100

bw, x  = 0.15, np.arange(len(metrics))
colors = ['#FFE975', '#4C72B0']
hatch  = ['//', '']

fig, ax = plt.subplots(figsize=(10, 5))

for j, (norm, raw, h, lbl, col) in enumerate(zip(
        [com_norm, sem_norm],
        [com_raw,  sem_raw],
        hatch,
        ['Virar (Com)', 'Virar (Sem)'],
        colors)):

    pos  = x + (j - 0.5) * bw
    bars = ax.bar(pos, norm, bw,
                  color=col, hatch=h,
                  edgecolor='black', linewidth=0.8, label=lbl)

    for bar, val in zip(bars, raw):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+2,
                f'{val:.2f}',
                ha='center', va='bottom',
                fontsize=8, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels([f'{m}\n(min=0, max={mx})'
                    for m, mx in zip(metrics, ranges_max)],
                   fontsize=9)

ax.set_ylim(0, max(com_norm.max(), sem_norm.max()) * 1.10)
ax.set_ylabel('Escala Normalizada (%)')
ax.set_title('Habilidade: Virar para o Obstáculo — Com vs. Sem Feedback', pad=15)
ax.yaxis.grid(True, ls='--', lw=0.5, alpha=0.7)

ax.legend(bbox_to_anchor=(0.5, 1.06), loc='lower center',
          ncol=2, frameon=False, fontsize=9)

plt.tight_layout()
plt.show()
