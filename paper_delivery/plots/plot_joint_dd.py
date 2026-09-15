"""Fig. 7 (joint_dd_v8): joint delay-Doppler power of the non-direct paths, LoS state, urban, overhead pass, 20 and 80 deg.
Data: paper_delivery/data/model/joint_v2_urban_e{20,80}.npz written by paper_delivery/evaluate/run_v2_all.py urban 3000 5.
Output: paper_delivery/figures/joint_dd_v8.pdf/.png and joint_dd_v8_audit.json (displayed power fraction, callout targets)."""
from pathlib import Path
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
from pathlib import Path
PD = Path(__file__).resolve().parents[1]; DM = PD/'data'/'model'; DR = PD/'data'/'references'; DS = PD/'data'/'scenemc'
FIG = PD/'figures'; FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.linewidth': .6, 'savefig.dpi': 400,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
BLUE = '#0072B2'; ORANGE = '#C66500'
ny = np.arange(-2.0, 2.0001, .02); yc = (ny[1:]+ny[:-1])/2
fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.6), sharey=True)
fig.subplots_adjust(left=.16, right=.98, bottom=.40, top=.90, wspace=.20)
audit = []

def contour_level(h, fraction=.9):
    s = np.sort(h.ravel())[::-1]; idx = np.searchsorted(np.cumsum(s), fraction*s.sum())
    return s[min(idx, len(s)-1)]

def callout(ax, text, xy, pos, color, style='-'):
    ax.annotate(text, xy, xytext=pos, color=color, fontsize=6.5,
                bbox=dict(boxstyle='round,pad=.20', fc='white', ec=color, lw=.6, alpha=.97),
                arrowprops=dict(arrowstyle='->', color=color, lw=.65, linestyle=style, connectionstyle='arc3,rad=.05'), zorder=15)

for col, e in enumerate((20, 80)):
    ax = axes[col]; r = np.load(DM/f'joint_v2_urban_e{e}.npz')
    tau, nu, w, mech = r['tau'], r['nu'], r['w'], r['mech']
    tx = np.arange(0, 600.0001, 2.5 if e == 20 else .125); xc = (tx[1:]+tx[:-1])/2
    total = float(w.sum())                                   # LoS-state non-direct power (ground + walls + edges)
    hist = {k: np.histogram2d(tau[mech == k]*1e9, nu[mech == k], bins=(tx, ny), weights=w[mech == k])[0].T/total for k in (1, 2)}
    h = hist[1]+hist[2]; db = 10*np.log10(np.maximum(h, 1e-99))
    im = ax.pcolormesh(tx, ny, np.ma.masked_where(db < -60, db), cmap='Greys', norm=Normalize(-60, -10), rasterized=True)
    for k, color, style in [(1, BLUE, 'solid'), (2, ORANGE, 'dashed')]:
        sm = gaussian_filter(hist[k], (.8, 1.2) if e == 20 else (1.6, 2.5))
        ax.contour(xc, yc, sm, levels=[contour_level(sm)], colors=[color], linewidths=[.85], linestyles=[style], zorder=8)
    g = mech == 3; gp = (float(tau[g][0]*1e9), float(nu[g][0]))
    ax.scatter(*gp, marker='^', facecolor='white', edgecolor='#111111', s=17, linewidth=.7, zorder=12)
    ax.set_xlim(0, 400 if e == 20 else 20); ax.set_ylim(-1.2, 1.2)
    ax.axhline(0, color='#666666', lw=.45, ls=':', zorder=1)
    ax.set_title(rf'$\beta_{{\rm ele}}={e}^\circ$', pad=5)
    ax.text(.985, .97, f'({"ab"[col]})', transform=ax.transAxes, ha='right', va='top', fontsize=6.5)
    ax.set_yticks([-1, 0, 1]); ax.set_xticks([0, 200, 400] if e == 20 else [0, 10, 20])
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    # arrow targets on the actual support: wall = weighted mean near a delay target, edge = strongest edge path in a window
    m = mech == 1; target = 75 if e == 20 else 2
    sel = m & (abs(tau*1e9-target) < (.5 if e == 20 else .05))
    refl = (float(np.average(tau[sel]*1e9, weights=w[sel])), float(np.average(nu[sel], weights=w[sel])))
    m = mech == 2; sel = m & (tau*1e9 > (10 if e == 20 else 1)) & (tau*1e9 < (100 if e == 20 else 8))
    ind = np.flatnonzero(sel)[np.argmax(w[sel])]; diffr = (float(tau[ind]*1e9), float(nu[ind]))
    callout(ax, 'Wall\nreflection', refl, (140 if e == 20 else 7, .67), BLUE)
    callout(ax, 'Roof-edge\ndiffraction', diffr, (125 if e == 20 else 6.5, -.96), ORANGE, '--')
    callout(ax, 'Ground', gp, (160 if e == 20 else 11.5, -.36), '#222222')
    inview = (tau*1e9 <= ax.get_xlim()[1]) & (abs(nu) <= 1.2)
    audit.append(dict(elevation=e, displayed_fraction=float(w[inview].sum()/total), refl=refl, diffr=diffr, ground=gp,
                      P_g=float(w[g].sum()/total), P_w=float(w[mech == 1].sum()/total), P_e=float(w[mech == 2].sum()/total),
                      below_floor=float(h[db < -60].sum()), n_paths=int(len(w))))
fig.text(.015, .65, r'Residual Doppler $\Delta\nu$ (Hz)', rotation=90, va='center', fontsize=7)
fig.text(.57, .265, r'Excess delay $\Delta\tau$ (ns)', ha='center', fontsize=7)
cax = fig.add_axes([.19, .16, .72, .025])
cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[-60, -40, -20, -10])
cb.set_label(r'Relative power per bin (dB)', labelpad=2, fontsize=7); cb.outline.set_linewidth(.5)
for ext in ('png', 'pdf'): fig.savefig(FIG/f'joint_dd_v8.{ext}', dpi=400, facecolor='white')
(FIG/'joint_dd_v8_audit.json').write_text(json.dumps(audit, indent=1))
print(json.dumps(audit, indent=1))
