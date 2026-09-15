"""Fig. 10 (cir_v2): one-scene path-wise TDL realization over 100 s from 20 deg and from 80 deg (3-D stems, one stem per
visible path: black LoS, blue wall reflection, orange roof-edge diffraction, grey ground reflection; power relative to
the free-space direct path, floor -40 dB, no fading drawn).
Data: paper_delivery/data/model/cir_seed{SEED}.json from paper_delivery/evaluate/cir_realization.py run SEED.  Output: paper_delivery/figures/cir_v2.pdf/.png
Usage: python3 paper_delivery/plots/plot_cir.py [SEED]
"""
import sys, json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
PD = Path(__file__).resolve().parents[1]; DM = PD/'data'/'model'
FIG = PD/'figures'; FIG.mkdir(exist_ok=True)
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 237      # paper realization
d = json.load(open(DM/f'cir_seed{SEED}.json'))
CW, CE, CG, CL = '#0072B2', '#D55E00', '0.45', '0.15'; COL = {1: CW, 2: CE, 3: CG}; FLOOR = -40.

# ---------------- Fig. 10: 3-D stems ----------------
fig = plt.figure(figsize=(7.4, 3.4))
for j, e in enumerate((20, 80)):
    ax = fig.add_subplot(1, 2, j+1, projection='3d')
    for r in d['windows'][f'e{e}']['records']:
        tt = r['t_rel']
        rows = ([(0.0, 1.0, CL)] if r['los'] else []) + [(tau, p, COL[m]) for m, tau, p in zip(r['mech'], r['tau_ns'], r['power'])]
        for tau, p, c in rows:
            z = 10*np.log10(max(p, 1e-9))
            if z < FLOOR: continue
            ax.plot([tt, tt], [tau, tau], [FLOOR, z], lw=0.9, color=c, alpha=0.95, solid_capstyle='butt')
            ax.plot([tt], [tau], [z], marker='o', ms=2.4, color=c, mec='none')
    ax.set_xlabel('time [s]', fontsize=7); ax.set_ylabel(r'$\Delta\tau$ [ns]', fontsize=7); ax.set_zlabel(r'$|h_i|^2$ [dB]', fontsize=7)
    ax.set_zlim(FLOOR, 2); ax.set_xlim(0, 100); ax.set_ylim(0, 600)
    ax.set_title(fr'$\beta_{{\rm ele}}(t{{=}}0)={e}^\circ$', fontsize=9)
    ax.tick_params(labelsize=6); ax.view_init(elev=24, azim=-72)
    ax.xaxis.labelpad = -4; ax.yaxis.labelpad = -2; ax.zaxis.labelpad = -2
fig.legend(handles=[Line2D([], [], color=CL, label='LoS'), Line2D([], [], color=CW, label='specular wall'),
                    Line2D([], [], color=CE, label='roof-edge diffraction'), Line2D([], [], color=CG, label='ground')],
           fontsize=6.5, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.savefig(FIG/'cir_v2.png', dpi=200, bbox_inches='tight'); fig.savefig(FIG/'cir_v2.pdf', bbox_inches='tight'); plt.close(fig)
print('wrote', FIG/'cir_v2.pdf')
