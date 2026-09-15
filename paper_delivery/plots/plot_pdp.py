"""Fig. 12 (fig11_pdp_state_0915): two-state normalized non-direct PDP at 20 and 80 deg, model (paper_delivery/evaluate/eval29.py) +
scene-MC markers (paper_delivery/evaluate/scene_pdp_state.py).  The submitted figure also carries dotted QuaDRiGa
(QuaDRiGa_NTN_*_LOS, 250 draws) curves; that data is not part of this release, so the overlay is omitted here.
Model PDPs merge every eval29 json listed in RUNS (per-city-normalized histograms weighted by city count).
Output: paper_delivery/figures/fig11_pdp_state_0915.pdf/.png"""
import json, glob, numpy as np, matplotlib
from pathlib import Path
PD = Path(__file__).resolve().parents[1]; DM = PD/'data'/'model'; DR = PD/'data'/'references'; DS = PD/'data'/'scenemc'
FIG = PD/'figures'; FIG.mkdir(exist_ok=True)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.4, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'font.family': 'serif', 'mathtext.fontset': 'dejavuserif', 'axes.grid': True, 'grid.alpha': 0.35, 'grid.linewidth': 0.5})
BLUE = (0.0, 0.447, 0.698); ORANGE = (0.835, 0.369, 0.0); GREEN = (0.0, 0.62, 0.45)
RUNS = {'urban': ['eval29_urban_n6000_s3.json', 'eval29_urban_n30000_s11.json', 'eval29_urban_n30000_s21.json'],
        'rural': ['eval29_rural_n6000_s4.json', 'eval29_rural_n30000_s12.json', 'eval29_rural_n30000_s22.json']}
ELEV = (20, 80)
EC = {20: BLUE, 80: ORANGE}

def model_pdp(scene, st, e):
    acc = None; n = 0
    for f in RUNS[scene]:
        try: d = json.load(open(DM/f))
        except FileNotFoundError: continue
        if f'{scene}_{st}_{e:g}' not in d['pdp']: continue
        p = d['pdp'][f'{scene}_{st}_{e:g}']; nc = d['meta']['ncity']
        y = np.array(p['p'])*nc; acc = y if acc is None else acc+y; n += nc; x = np.array(p['centers_ns'])
    return x, acc/n, n

fig, axs = plt.subplots(2, 1, figsize=(3.5, 4.0)); ncity = {}
for ax, scene in zip(axs, ('urban', 'rural')):
    for e in ELEV:
      if True:
        for st, ls in (('los', '-'), ('nlos', '--')):
            x, y, ncity[scene] = model_pdp(scene, st, e); m = y > 0
            ax.loglog(x[m], y[m], ls, color=EC[e], lw=1.3, zorder=3)
            mcf = sorted(glob.glob(str(DS/f'scenemc_state_{scene}_e{e}_n*_s*.json')))
            if mcf:
                mc = json.load(open(mcf[-1]))
                if st in mc:
                    ce = np.array(mc['edges']); cx = 0.5*(ce[1:]+ce[:-1]); cy = np.array(mc[st]['pdp'])
                    pick = np.unique([np.argmin(abs(cx-v)) for v in np.geomspace(cx[0], 3300, 16)]); pick = pick[cy[pick] > 0]
                    ax.plot(cx[pick], cy[pick], 'o', color=EC[e], ms=3.4, mfc=EC[e] if st == 'los' else 'white', mew=0.9, ls='none', zorder=5)
    ax.set_xlim(10, 10000); ax.set_ylim(1e-7, 2); ax.set_ylabel('Normalized non-direct PDP')
    ax.set_xticks([10, 33, 100, 330, 1000, 3300, 10000]); ax.set_xticklabels(['10', '33', '100', '330', '1000', '3300', '10000']); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.text(0.03, 0.04, '(a) Urban' if scene == 'urban' else '(b) Rural', transform=ax.transAxes, va='bottom', fontsize=8)
axs[1].set_xlabel('Excess delay (ns)')
h = [Line2D([], [], color=EC[e], lw=1.3, label=f'{e}°') for e in ELEV] + \
    [Line2D([], [], color='k', lw=1.3, ls='-', label='LoS state'), Line2D([], [], color='k', lw=1.3, ls='--', label='NLoS state'),
     Line2D([], [], color='k', marker='o', ms=3.4, ls='none', label='scene MC, LoS'), Line2D([], [], color='k', marker='o', ms=3.4, mfc='white', ls='none', label='scene MC, NLoS')]
axs[0].legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, handlelength=1.8, columnspacing=0.9)
fig.tight_layout(pad=0.35, h_pad=0.5); fig.savefig(FIG/'fig11_pdp_state_0915.pdf'); fig.savefig(FIG/'fig11_pdp_state_0915.png', dpi=220)
print('cities merged:', ncity)
