"""Fig. 13 (fig12_ds_state_0914): RMS delay spread per state vs elevation; Fig. 15 (fig14_doppler_state_0914): RMS Doppler
spread and RMS coherence time per state vs elevation.  Model: paper_delivery/data/model/eval29_urban_n6000_s3.json and eval29_rural_n6000_s4.json
(paper_delivery/evaluate/eval29.py).  References: paper_delivery/data/references/fig13_data.csv (TR 38.811 LOS DS rows, 6G-NTN D4.1 digitized, QuaDRiGa
configured DS means), TR 38.811 NLOS DS rows inline, paper_delivery/data/references/wcl_bound_paper_plot_data.csv (QuaDRiGa sigma_nu, WCL
scatterer-only bound).  Outputs: paper_delivery/figures/fig12_ds_state_0914.pdf, paper_delivery/figures/fig14_doppler_state_0914.pdf (+png).
"""
import json, csv, numpy as np, matplotlib
from pathlib import Path
PD = Path(__file__).resolve().parents[1]; DM = PD/'data'/'model'; DR = PD/'data'/'references'; DS = PD/'data'/'scenemc'
FIG = PD/'figures'; FIG.mkdir(exist_ok=True)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.4,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'font.family': 'serif', 'mathtext.fontset': 'dejavuserif',
                     'axes.grid': True, 'grid.alpha': 0.35, 'grid.linewidth': 0.5})
BLUE = (0.0, 0.447, 0.698); ORANGE = (0.835, 0.369, 0.0); GREEN = (0.0, 0.62, 0.45)
G1, G2, G3 = '#6e6e6e', '#9a9a9a', '#4b4b4b'
E = {s: json.load(open(DM/f)) for s, f in (('urban', 'eval29_urban_n6000_s3.json'), ('rural', 'eval29_rural_n6000_s4.json'))}
d = {'res': {s: E[s]['res'] for s in E}}; B = np.array(E['urban']['meta']['beta'])
R = {s: {k: np.array([r[k] for r in d['res'][s]]) for k in d['res'][s][0]} for s in ('urban', 'rural')}
XL = r'Elevation $\beta_{\rm ele}$ (deg)'

def ystyle(ax, ylabel, ylim, tk):
    ax.set_yscale('log'); ax.set_ylabel(ylabel); ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(FixedLocator(tk)); ax.yaxis.set_major_formatter(FixedFormatter([str(v) for v in tk]))
    ax.yaxis.set_minor_formatter(NullFormatter())
def xstyle(ax, label=True):
    ax.set_xlim(8, 82); ax.set_xticks([10, 20, 30, 40, 50, 60, 70, 80])
    if label: ax.set_xlabel(XL)
def model_pair(ax, x, yu, yr, state):
    ls = '-' if state == 'los' else '--'; mfc = None if state == 'los' else 'none'
    ax.plot(x, yu, ls, marker='o', color=BLUE, lw=1.4, ms=3.5, mfc=mfc, label=f'Urban, {"LoS" if state=="los" else "NLoS"}')
    ax.plot(x, yr, ls, marker='s', color=ORANGE, lw=1.4, ms=3.5, mfc=mfc, label=f'Rural, {"LoS" if state=="los" else "NLoS"}')
def ref(ax, x, y, label, color, mk, ls=':'):
    ax.plot(x, y, ls, color=color, lw=1.0, marker=mk, ms=3.0, mfc='none', label=label)
def tag(ax, t): ax.text(0.02, 0.97, t, transform=ax.transAxes, va='top', fontsize=8)
def leg(ax, **kw):
    a = dict(framealpha=0.92, handlelength=2.2, borderpad=0.35, labelspacing=0.25, columnspacing=0.8); a.update(kw)
    ax.legend(**a)

# ================= Fig. 12: RMS delay spread =================
E8 = np.arange(10, 81, 10)
f13 = list(csv.DictReader(open(DR/'fig13_data.csv')))
def refds(src, scn): return 10**np.array([float(r['value']) for r in f13 if r['source'] == src and r['scenario'] == scn])*1e9
NLOS_38811 = {'urban': [-7.24, -7.70, -7.82, -8.04, -8.08, -8.10, -8.16, -8.03],
              'dense urban': [-6.84, -6.81, -6.94, -7.14, -7.34, -7.53, -7.67, -7.82],
              'rural': [-9.01, -8.37, -8.05, -7.92, -7.92, -7.96, -7.91, -7.79]}
fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 4.3))
L10 = lambda ns: np.log10(np.asarray(ns, float)*1e-9)
model_pair(a1, B, L10(R['urban']['ds_los_ns']), L10(R['rural']['ds_los_ns']), 'los')
ref(a1, E8, L10(refds('3GPP TR 38.811', 'urban')), 'TR 38.811 LOS, urban', G1, '^')
ref(a1, E8, L10(refds('3GPP TR 38.811', 'dense urban')), 'TR 38.811 LOS, dense urban', G1, 'v')
ref(a1, E8, L10(refds('3GPP TR 38.811', 'rural')), 'TR 38.811 LOS, rural', G1, 'd')
ref(a1, E8, L10(refds('6G-NTN', 'urban')), '6G-NTN, urban (3.5 GHz)', GREEN, 'x', '-.')
ref(a1, E8, L10(refds('QuaDRiGa', 'urban')), 'QuaDRiGa LOS, urban', G3, 'D', '-.')
ref(a1, E8, L10(refds('QuaDRiGa', 'rural')), 'QuaDRiGa LOS, rural', G3, 'p', '-.')
xstyle(a1, False); a1.set_yscale('linear'); a1.set_ylim(-9.7, -6.4); a1.set_yticks(np.arange(-9.5, -6.4, 0.5)); a1.set_ylabel(r'$\log_{10}(\mathrm{RMS\ DS}/1\,\mathrm{s})$'); tag(a1, '(a) LoS state')
leg(a1, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
model_pair(a2, B, L10(R['urban']['ds_nlos_ns']), L10(R['rural']['ds_nlos_ns']), 'nlos')
ref(a2, E8, np.array(NLOS_38811['urban']), 'TR 38.811 NLOS, urban', G1, '^')
ref(a2, E8, np.array(NLOS_38811['dense urban']), 'TR 38.811 NLOS, dense urban', G1, 'v')
ref(a2, E8, np.array(NLOS_38811['rural']), 'TR 38.811 NLOS, rural', G1, 'd')
xstyle(a2); a2.set_yscale('linear'); a2.set_ylim(-9.2, -6.4); a2.set_yticks(np.arange(-9.0, -6.4, 0.5)); a2.set_ylabel(r'$\log_{10}(\mathrm{RMS\ DS}/1\,\mathrm{s})$'); tag(a2, '(b) NLoS state')
leg(a2, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
fig.tight_layout(pad=0.35, h_pad=0.5); fig.savefig(FIG/'fig12_ds_state_0914.pdf'); fig.savefig(FIG/'fig12_ds_state_0914.png', dpi=220); plt.close(fig)

# ================= Fig. 14: sigma_nu and T_c =================
w = list(csv.DictReader(open(DR/'wcl_bound_paper_plot_data.csv')))
def refnu(src, scn):
    rr = sorted([r for r in w if r['source'] == src and r['scene'] == scn], key=lambda r: float(r['elevation_deg']))
    return np.array([float(r['elevation_deg']) for r in rr]), np.array([float(r['rms_Hz']) for r in rr])
qu, qr, wc = refnu('QuaDRiGa', 'urban'), refnu('QuaDRiGa', 'rural'), refnu('WCL', 'hemisphere')
fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 5.4))
for ax, key, yl, ylim, tk in ((a1, 'dsp', r'RMS Doppler spread $\sigma_\nu$ (Hz)', (0.015, 200), [0.03, 0.1, 0.3, 1, 3, 10, 30, 100]),
                              (a2, 'Tc', r'$T_c^{\rm RMS}=1/(2\pi\sigma_\nu)$ (s)', (0.0007, 30), [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])):
    f = (lambda y: y) if key == 'dsp' else (lambda y: 1/(2*np.pi*y))
    model_pair(ax, B, R['urban'][f'{key}_los_Hz' if key == 'dsp' else 'Tc_los_s'], R['rural'][f'{key}_los_Hz' if key == 'dsp' else 'Tc_los_s'], 'los')
    model_pair(ax, B, R['urban'][f'{key}_nlos_Hz' if key == 'dsp' else 'Tc_nlos_s'], R['rural'][f'{key}_nlos_Hz' if key == 'dsp' else 'Tc_nlos_s'], 'nlos')
    ref(ax, qu[0], f(qu[1]), 'QuaDRiGa LOS, urban', G3, 'D', '-.'); ref(ax, qr[0], f(qr[1]), 'QuaDRiGa LOS, rural', G3, 'p', '-.')
    ref(ax, wc[0], f(wc[1]), 'Upper bound [14]', G1, 'x', ':')   # WCL scatterer-only bound; [14] = its citation number in the manuscript
    xstyle(ax, ax is a2); ystyle(ax, yl, ylim, tk)
tag(a1, '(a)'); tag(a2, '(b)')
leg(a1, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
fig.tight_layout(pad=0.35, h_pad=0.5); fig.savefig(FIG/'fig14_doppler_state_0914.pdf'); fig.savefig(FIG/'fig14_doppler_state_0914.png', dpi=220); plt.close(fig)

