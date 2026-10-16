"""Example: the blocking-wall diffraction (mechanism 4 of model/physical_model.py).

When a wall blocks the direct path, the terminal still receives the signal that diffracts over that wall's roof edge
into its shadow.  The model computes it for every wall that blocks the direct path: exact stationary point on the
roof edge, single knife-edge loss J(v) with v = 2 sqrt(excess path / lambda) (ITU-R P.526).  It exists only in the
NLoS state; roof-edge diffraction of mechanism 2 covers walls with the satellite and the terminal on the same side.

What it plots
  (a) One 10 m wall across the satellite azimuth, the terminal at distance d behind it: direct path (LoS side) and
      blocking-wall diffraction (shadow side) for 20, 40 and 80 deg.  At the shadow boundary the diffracted power is
      -6 dB, the knife-edge value at v = 0; deeper into the shadow it falls off.
  (b) The urban scene of Fig. 10 (seed 237) from 80 deg: non-direct power per mechanism every second with ray-traced
      visibility (model/tdl.py).  After the LoS blockage at t = 28 s the blocking wall's diffraction carries the link.

Usage: python3 examples/blocking_wall_diffraction.py      -> examples/blocking_wall_diffraction.png (about 20 s)
"""
import sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'model'))
import physical_model as pm
from tdl import TDLChannel

H_WALL = 10.0
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))

# (a) one wall, terminal moved through its shadow
M = pm.Model('urban')
for e, c in ((20, '#0072B2'), (40, '#D55E00'), (80, '#009E73')):
    t = pm.reference_time(e); s = pm.satellite(t); az = np.arctan2(s[1], s[0])
    shadow = (H_WALL - pm.UE[2])/np.tan(np.radians(e))                 # depth of the wall's shadow
    D = np.linspace(0.2, 2*shadow, 300); pd = np.full(len(D), np.nan); pb = np.full(len(D), np.nan)
    for i, d in enumerate(D):
        city = dict(x=np.array([d*np.cos(az)]), y=np.array([d*np.sin(az)]), angle=np.array([az+np.pi/2]),
                    height=np.array([H_WALL]), weight=np.ones(1))
        p = M.paths(t, city, geometric_blockage=True); k = p['mechanism'] == 4
        if p['visible'][-1]: pd[i] = 0.0
        if k.any(): pb[i] = 10*np.log10(p['amp'][k].sum())
    ax[0].plot(D/shadow, pb, color=c, lw=1.8, label=f'{e}°, over the wall (shadow {shadow:.1f} m)')
    ax[0].plot(D/shadow, pd, color=c, lw=1.0, ls='--')
ax[0].axvline(1, color='0.5', lw=0.8); ax[0].axhline(-6, color='0.5', lw=0.8, ls=':')
ax[0].text(1.03, -16, 'shadow\nboundary', fontsize=7, color='0.4'); ax[0].text(1.55, -5.2, '-6 dB', fontsize=7, color='0.4')
ax[0].set_xlabel('distance behind the wall / shadow depth'); ax[0].set_ylabel('power re. free-space direct path [dB]')
ax[0].set_ylim(-40, 3); ax[0].set_title(f'(a) one {H_WALL:g} m wall; dashed: direct path', fontsize=9)
ax[0].legend(fontsize=7, loc='lower right')

# (b) Fig. 10 scene from 80 deg, ray-traced visibility
ch = TDLChannel('urban', seed=237); t0 = ch.time_at_elevation(80.); T = np.arange(0, 100.01, 1.0)
NAMES = {3: ('ground', '0.45'), 1: ('wall reflection', '#0072B2'), 2: ('roof-edge diffraction', '#D55E00'),
         4: ('over the blocking wall', '#009E73')}
P = {m: np.full(len(T), np.nan) for m in NAMES}; los = np.zeros(len(T), bool)
for i, u in enumerate(T):
    q = ch.paths(t0+u); los[i] = q['los']
    for m in NAMES:
        k = q['mechanism'] == m
        if k.any(): P[m][i] = 10*np.log10(q['power'][k].sum())
for m, (name, c) in NAMES.items():
    ax[1].plot(T, P[m], 'o-', ms=2.5, lw=1.0, color=c, label=name)
ax[1].fill_between(T, -60, 5, where=~los, color='0.85', lw=0, label='LoS blocked')
ax[1].set_ylim(-60, 3); ax[1].set_xlim(0, 100); ax[1].set_xlabel('time [s] (satellite from 80°, through zenith)')
ax[1].set_ylabel('non-direct power per mechanism [dB]'); ax[1].legend(fontsize=7, loc='lower left')
ax[1].set_title('(b) urban scene 237 (Fig. 10), ray-traced visibility', fontsize=9)
for a in ax: a.grid(alpha=0.3)
fig.tight_layout(); out = ROOT/'examples'/'blocking_wall_diffraction.png'; fig.savefig(out, dpi=150); print('wrote', out)
first = T[~los][0] if (~los).any() else None
print(f'scene 237 from 80 deg: LoS blocked from t = {first} s; blocking-wall diffraction while blocked: '
      f'{np.nanmin(P[4][~los]):.1f} to {np.nanmax(P[4][~los]):.1f} dB; other non-direct paths at most '
      f'{np.nanmax([np.nanmax(P[m][~los]) if np.isfinite(P[m][~los]).any() else -np.inf for m in (1, 2, 3)]):.1f} dB')
