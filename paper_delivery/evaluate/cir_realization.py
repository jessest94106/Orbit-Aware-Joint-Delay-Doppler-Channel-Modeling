"""One-scene TDL realization of eq. (hnlos) on the drawn-city geometry (Fig. 10).

One urban city is drawn once (Model.draw_city, seed SEED) and frozen; the satellite moves on the overhead
550 km pass.  At every time sample all wall / roof-edge / ground paths are rebuilt from scratch with
Model.paths(t, city, geometric_blockage=True), so path continuity is a property of the geometry, not of any
tracker.  The direct path is ray-tested against every wall (visible[-1]); the ground reflection carries the same
indicator (Sec. IV, eq. existence); walls and roof edges use their own ray tests.  Powers are relative to the
direct path in free space (direct path = 0 dB); no fading is drawn.

Outputs data/model/cir_seed{SEED}.json with
  windows   : 'e20' and 'e80' = 100 s from beta_ele = 20 deg (ascending) and from 80 deg (through zenith,
              descending), sampled every 4 s: per sample t, beta_deg, los (bool) and the visible paths
              (mechanism 1 wall / 2 roof edge / 3 ground, tau_ns, power, owner = index of the wall that owns the path).
Usage:
  python3 paper_delivery/evaluate/cir_realization.py run SEED        (paper seed 237; ~30 s)
  python3 paper_delivery/evaluate/cir_realization.py search N0 N1    (score seeds N0..N1-1 for the selection rule below)
Selection rule used for the paper figure: LoS clear during the whole 20-deg window, at least one wall path
visible at the start of the 20-deg window (the expected number of valid wall paths at 20 deg is only ~0.5), and a LoS blockage inside the 80-deg window on the descending
side (t > 20 s) caused by a wall at least 3 m from the terminal, so that the figure shows the indicator
switching of eq. (existence) once.
"""
import sys, json, time, numpy as np
from pathlib import Path
PD = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(PD.parent/'model')); OUT = PD/'data'/'model'
import physical_model as pm
SCENE = 'urban'; WIN = 100.0; DT_STEM = 4.0

def blocking_walls(M, end, city):
    """Indices of the walls that intercept the finite ray terminal -> end (same test as Model.clear)."""
    ray = end-pm.UE; e = np.array([np.cos(city['angle']), np.sin(city['angle'])]); centers = np.array([city['x'], city['y']])
    den = pm.cross(ray[:2, None], e)
    t = np.divide(pm.cross(centers, e), den, out=np.full(len(den), np.inf), where=abs(den) > 1e-12)
    along = np.divide(pm.cross(centers, ray[:2, None]), den, out=np.full(len(den), np.inf), where=abs(den) > 1e-12)
    z = pm.UE[2]+t*ray[2]
    return np.flatnonzero((t > 1e-9) & (t < 1-1e-9) & (abs(along) <= M.width/2) & (z >= 0) & (city['height'] > z))

def sample(M, city, t):
    p = M.paths(t, city, geometric_blockage=True); los = bool(p['visible'][-1])
    keep = p['visible'].copy(); keep[-1] = los                        # ground shares the LoS indicator
    rec = dict(t=float(t), beta_deg=float(np.degrees(pm.elevation(t))), los=los,
               mech=p['mechanism'][keep].tolist(), tau_ns=(p['tau'][keep]*1e9).tolist(), power=p['amp'][keep].tolist(),
               owner=p['owner'][keep].tolist())
    return rec

def realization(seed):
    M = pm.Model(SCENE); city = M.draw_city(np.random.default_rng(seed))
    out = dict(meta=dict(scene=SCENE, seed=seed, n_walls=int(len(city['x'])), fc_GHz=pm.FC/1e9, pass_='overhead 550 km',
                         window_s=WIN, dt_stem_s=DT_STEM), windows={})
    for e in (20., 80.):
        t0 = pm.reference_time(e); recs = []
        for u in np.arange(0, WIN+1e-9, DT_STEM):
            r = sample(M, city, t0+u); r['t_rel'] = float(u); recs.append(r)
        out['windows'][f'e{e:g}'] = dict(t0=float(t0), records=recs)
    return M, city, out

def score(seed):
    M = pm.Model(SCENE); city = M.draw_city(np.random.default_rng(seed))
    t20 = pm.reference_time(20.); t80 = pm.reference_time(80.)
    r20 = [sample(M, city, t20+u) for u in np.arange(0, WIN+1e-9, DT_STEM)]; los20 = [r['los'] for r in r20]
    nw = sum(1 for m in r20[0]['mech'] if m == 1)
    walls20 = len({o for r in r20 for o, m in zip(r['owner'], r['mech']) if m == 1})     # distinct wall paths over the window
    block = None; dist = None
    for u in np.arange(0, WIN+1e-9, DT_STEM):
        s = pm.satellite(t80+u)
        if not M.clear(s, city):
            i = blocking_walls(M, s, city); dist = float(np.hypot(city['x'][i], city['y'][i]).min()); block = float(u); break
    return dict(seed=seed, los20_clear=all(los20), n_wall_20=nw, n_walls_20_window=walls20, block80_t=block, block_wall_dist_m=dist)

if __name__ == '__main__':
    if sys.argv[1] == 'search':
        for sd in range(int(sys.argv[2]), int(sys.argv[3])):
            sc = score(sd); ok = sc['los20_clear'] and sc['n_wall_20'] >= 1 and sc['block80_t'] is not None and sc['block80_t'] > 20 and sc['block_wall_dist_m'] >= 3
            print(json.dumps(sc), 'PASS' if ok else '', flush=True)
    else:
        seed = int(sys.argv[2]); t = time.time(); M, city, out = realization(seed)
        s80 = out['windows']['e80']['records']; blk = [r['t_rel'] for r in s80 if not r['los']]
        out['meta']['los_block_80_first_s'] = blk[0] if blk else None
        json.dump(out, open(OUT/f'cir_seed{seed}.json', 'w'))
        print(f"seed {seed}: {out['meta']['n_walls']} walls; 20 deg window LoS clear {all(r['los'] for r in out['windows']['e20']['records'])}, "
              f"walls at t=0: {sum(1 for m in out['windows']['e20']['records'][0]['mech'] if m==1)}; 80 deg window first LoS block at t = {blk[0] if blk else None} s; "
              f"{time.time()-t:.0f} s")
