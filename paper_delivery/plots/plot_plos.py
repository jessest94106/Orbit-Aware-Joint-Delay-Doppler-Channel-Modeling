"""Fig. 9 (fig_plos_state_0915): LoS / NLoS state probability vs elevation, urban and rural, from the closed form
P_LoS = E[1_0] = exp(-lambda_1D * tail(h_g) / tan(beta)), tail(h) = gamma sqrt(pi/2) erfc(h / (sqrt 2 gamma))  (Sec. IV, eq. blockage_B),
with the TR 38.811 Table 6.6.1-1 LOS rows.  No simulation data needed.  Outputs paper_delivery/figures/fig_plos_state_0915.pdf and plos_0915.csv."""
import sys, json, csv, numpy as np, matplotlib
from pathlib import Path
PD = Path(__file__).resolve().parents[1]; DM = PD/'data'/'model'; DR = PD/'data'/'references'; DS = PD/'data'/'scenemc'
FIG = PD/'figures'; FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(PD.parent/'model')); import physical_model as pm
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.6, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'font.family': 'serif', 'mathtext.fontset': 'dejavuserif', 'axes.grid': True, 'grid.alpha': 0.35, 'grid.linewidth': 0.5})
BLUE = (0.0, 0.447, 0.698); ORANGE = (0.835, 0.369, 0.0); G1, G3 = '#6e6e6e', '#4b4b4b'
B = np.arange(5, 90.01, 1.0)
P = {}
for scene in ('urban', 'rural'):
    M = pm.Model(scene)
    P[scene] = np.exp(-M.lambda1d*M.tail(pm.UE[2])/np.tan(np.radians(B)))
E9 = np.arange(10, 91, 10)
TR = {'dense urban': [.282, .331, .398, .468, .537, .612, .738, .820, .981],
      'urban':       [.246, .386, .493, .613, .726, .805, .919, .968, .992],
      'suburban/rural': [.782, .869, .919, .929, .935, .940, .949, .952, .998]}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.5, 2.3))
a1.plot(B, P['urban'], '-', color=BLUE, lw=1.6, label='Urban')
a1.plot(B, P['rural'], '-', color=ORANGE, lw=1.6, label='Rural')
for k, mk in (('urban', 'o'), ('dense urban', '^'), ('suburban/rural', 's')):
    a1.plot(E9, TR[k], ':', color=G1, lw=1.0, marker=mk, ms=3.2, mfc='none', label=f'TR 38.811, {k}')
a1.set_ylim(0, 1.02); a1.set_ylabel(r'$P_{\rm LoS}=\mathbb{E}[\mathbf{1}_0]$'); a1.set_yticks(np.arange(0, 1.01, 0.2))
h, l = a1.get_legend_handles_labels()
fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=6.4, handlelength=1.8, columnspacing=0.8)
a1.set_title('(a) LoS', fontsize=8, pad=3)
a2.semilogy(B, 1-P['urban'], '-', color=BLUE, lw=1.6, label='Urban')
a2.semilogy(B, 1-P['rural'], '-', color=ORANGE, lw=1.6, label='Rural')
for k, mk in (('urban', 'o'), ('dense urban', '^'), ('suburban/rural', 's')):
    a2.semilogy(E9, 1-np.array(TR[k]), ':', color=G1, lw=1.0, marker=mk, ms=3.2, mfc='none', label=f'TR 38.811, {k}')
a2.set_ylim(5e-5, 1.2); a2.set_ylabel(r'$P_{\rm NLoS}=1-\mathbb{E}[\mathbf{1}_0]$')
tk = [1e-4, 1e-3, 1e-2, 1e-1, 1]
a2.yaxis.set_major_locator(FixedLocator(tk)); a2.yaxis.set_major_formatter(FixedFormatter(['0.0001', '0.001', '0.01', '0.1', '1']))
a2.yaxis.set_minor_formatter(NullFormatter())
a2.set_title('(b) NLoS', fontsize=8, pad=3)
a1.set_ylabel(r'$P_{\rm LoS}$', labelpad=1); a2.set_ylabel(r'$P_{\rm NLoS}$', labelpad=1)
for ax in (a1, a2):
    ax.set_xlim(4, 91); ax.set_xticks([10, 30, 50, 70, 90]); ax.tick_params(labelsize=7.5)
fig.text(0.56, 0.035, r'Elevation $\beta_{\rm ele}$ (deg)', ha='center', fontsize=8)
fig.subplots_adjust(left=0.12, right=0.99, top=0.76, bottom=0.23, wspace=0.55); fig.savefig(FIG/'fig_plos_state_0915.pdf'); fig.savefig(FIG/'fig_plos_state_0915.png', dpi=220)
with open(FIG/'plos_0915.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['beta_deg', 'P_LoS_urban', 'P_LoS_rural'])
    for b, u, r in zip(B, P['urban'], P['rural']): w.writerow([b, u, r])
for scene in ('urban', 'rural'):
    print(scene, 'P_LoS 10:10:90', ' '.join(f'{P[scene][np.argmin(abs(B-e))]:.4f}' for e in E9))
