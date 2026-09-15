"""Fig. 8 (fig_power_path_0915): LoS-state contributions of ground / wall / roof edge to P_nd^L and to the expected
valid-path count, urban + rural, 10..89 deg, shaded 85-90 deg.   Fig. 14 (fig13_k_0915): LoS-state K-factor, omni
urban/rural and urban 20-deg tracking beam, with the published references.
Data: paper_delivery/data/model/v2all_urban_n3000_s5.json, v2all_rural_n3000_s6.json (paper_delivery/evaluate/run_v2_all.py).
Reference rows: TR 38.811 Tables 6.7.2-1a/3a (S band, LOS K), 6G-NTN D4.1 Fig. 44 (3.5 GHz, digitized), QuaDRiGa v2.8.1
QuaDRiGa_NTN_{Urban,Rural}_LOS configured K means at 2 GHz (quadriga/qd_run_all.m + reduce_qd.py).
Outputs: paper_delivery/figures/fig_power_path_0915.pdf, paper_delivery/figures/fig13_k_0915.pdf, paper_delivery/figures/fig8_fig13_numbers_0915.csv."""
import json, numpy as np, matplotlib
from pathlib import Path
PD = Path(__file__).resolve().parents[1]; DM = PD/'data'/'model'; DR = PD/'data'/'references'; DS = PD/'data'/'scenemc'
FIG = PD/'figures'; FIG.mkdir(exist_ok=True)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.4, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'font.family': 'serif', 'mathtext.fontset': 'dejavuserif', 'axes.grid': True, 'grid.alpha': 0.35,
                     'grid.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
BLUE = (0.0, 0.447, 0.698); ORANGE = (0.835, 0.369, 0.0); GREEN = (0.0, 0.62, 0.45); PURPLE = '#7846A3'; PINK = '#CC79A7'; GREY = '#4d4d4d'
V = {s: json.load(open(DM/f'v2all_{s}_n3000_s{n}.json'))['res'] for s, n in (('urban', 5), ('rural', 6))}
B = np.array([r['beta'] for r in V['urban']])
def contrib(rows, keys):
    a = np.array([[r[k] for k in keys] for r in rows]); return 100*a/a.sum(1, keepdims=True)
XL = r'Elevation $\beta_{\rm ele}$ (deg)'

# ---------------- Fig. 8 ----------------
fig, axs = plt.subplots(2, 2, figsize=(3.5, 3.6), sharex=True, sharey=True)
STY = [('Ground reflection', GREY, '-'), ('Wall reflection', BLUE, '-'), ('Roof-edge diffraction', ORANGE, '-')]
for col, s in enumerate(('urban', 'rural')):
    Pc = contrib(V[s], ('Pg', 'Pw', 'Pe')); Nc = contrib(V[s], ('n_ground', 'n_wall', 'n_edge'))
    for row, Y in enumerate((Pc, Nc)):
        ax = axs[row, col]
        for j, (lab, c, ls) in enumerate(STY): ax.plot(B, Y[:, j], ls, color=c, lw=1.3, label=lab)
        ax.axvspan(85, 90, color='#d8dade', alpha=.6, zorder=-2, lw=0)
        ax.set_xlim(8, 90); ax.set_xticks(range(10, 91, 20)); ax.set_ylim(-2, 104); ax.set_yticks(range(0, 101, 25))
        ax.text(0.03, 0.95, f'({"abcd"[2*row+col]}) {s.capitalize()}', transform=ax.transAxes, va='top', fontsize=7.5)
axs[0, 0].set_ylabel(r'Contribution to $P_{\rm nd}^{\rm L}$ (%)'); axs[1, 0].set_ylabel('Valid-path contribution (%)')
for ax in axs[1]: ax.set_xlabel(XL)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', bbox_to_anchor=(.5, 1.0), ncol=3, frameon=False, fontsize=6.4, handlelength=1.8, columnspacing=1.0)
fig.subplots_adjust(left=.15, right=.98, top=.91, bottom=.11, hspace=.14, wspace=.10)
for ext in ('pdf', 'png'): fig.savefig(FIG/f'fig_power_path_0915.{ext}', dpi=300, facecolor='white')
plt.close(fig)

# ---------------- Fig. 13 ----------------
E = np.array([10, 20, 30, 40, 50, 60, 70, 80], float)
K811_URBAN = [31.83, 18.78, 10.49, 7.46, 6.52, 5.47, 4.54, 4.03]; K811_DURBAN = [4.40, 9.00, 9.30, 7.90, 7.40, 7.00, 6.90, 6.50]
K811_RURAL = [24.72, 12.31, 8.05, 6.21, 5.04, 4.42, 3.92, 3.65]; D41_URBAN = [4.2, 4.6, 6.7, 8.2, 6.5, 9.1, 11.8, 13.2]
QD_URBAN = [33.267, 29.956, 28.019, 26.645, 25.579, 24.708, 23.971, 23.333]; QD_RURAL = [22.135, 19.637, 18.175, 17.138, 16.334, 15.677, 15.121, 14.640]
m = B <= 80; Bm = B[m]
Ku = np.array([r['K_omni_dB'] for r in V['urban']])[m]; Kr = np.array([r['K_omni_dB'] for r in V['rural']])[m]; Kb = np.array([r['K_beam_dB'] for r in V['urban']])[m]
fig, ax = plt.subplots(figsize=(3.5, 3.0))
MOD = dict(lw=1.6, ms=3.5); REF = dict(lw=0.9, ms=2.8, mfc='none')
ax.plot(Bm, Ku, 'o-', color=BLUE, label='Model, urban (omni)', zorder=5, **MOD)
ax.plot(Bm, Kr, 's--', color=ORANGE, label='Model, rural (omni)', zorder=5, **MOD)
ax.plot(Bm, Kb, '^-.', color=PINK, label=r'Model, urban ($20^\circ$ tracking beam)', zorder=5, **MOD)
ax.plot(E, K811_URBAN, 's:', color='#6e6e6e',label='TR 38.811, urban', **REF)
ax.plot(E, K811_DURBAN, 'D:', color='#4b4b4b', label='TR 38.811, dense urban', **REF)
ax.plot(E, K811_RURAL, 'v:', color='#9a9a9a', label='TR 38.811, rural', **REF)
ax.plot(E, D41_URBAN, '^:', color=GREEN, label='6G-NTN, urban', **REF)
ax.plot(E, QD_URBAN, 'x:', color=PURPLE, label='QuaDRiGa, urban', **REF)
ax.plot(E, QD_RURAL, '+:', color=PURPLE, label='QuaDRiGa, rural', **REF)
ax.set_xlim(8, 82); ax.set_xticks(range(10, 81, 10)); ax.set_ylim(0, 40); ax.set_yticks(range(0, 41, 10))
ax.set_xlabel(XL); ax.set_ylabel('K-factor (dB)')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, columnspacing=0.8, labelspacing=0.3, handlelength=2.0)
fig.subplots_adjust(left=.13, right=.98, top=.76, bottom=.14)
for ext in ('pdf', 'png'): fig.savefig(FIG/f'fig13_k_0915.{ext}', dpi=300, facecolor='white')
with open(FIG/'fig8_fig13_numbers_0915.csv', 'w') as f:
    f.write('scene,beta,Pg_pct,Pw_pct,Pe_pct,ng_pct,nw_pct,ne_pct,n_total,K_omni_dB,K_beam_dB\n')
    for s in ('urban', 'rural'):
        Pc = contrib(V[s], ('Pg', 'Pw', 'Pe')); Nc = contrib(V[s], ('n_ground', 'n_wall', 'n_edge'))
        for i, r in enumerate(V[s]):
            f.write(f"{s},{r['beta']:g},{Pc[i,0]:.2f},{Pc[i,1]:.2f},{Pc[i,2]:.2f},{Nc[i,0]:.2f},{Nc[i,1]:.2f},{Nc[i,2]:.2f},{1+r['n_wall']+r['n_edge']:.3f},{r['K_omni_dB']:.2f},{r['K_beam_dB']:.2f}\n")
print(open(FIG/'fig8_fig13_numbers_0915.csv').read())
