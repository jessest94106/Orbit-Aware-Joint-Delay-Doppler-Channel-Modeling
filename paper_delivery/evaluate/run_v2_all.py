"""LoS-state mechanism powers, valid-path counts and K-factor vs elevation (eq. 29 evaluated on drawn cities),
plus the per-path (tau, nu, w) dump at 20/80 deg (urban) for the joint delay-Doppler figure.
Outputs data/model/v2all_SCENE_nNCITY_sSEED.json and data/model/joint_v2_urban_e{20,80}.npz.
Usage: python3 paper_delivery/evaluate/run_v2_all.py SCENE NCITY SEED   (paper: urban 3000 5, rural 3000 6; ~25 min each on one core)"""
import sys, json, time, numpy as np
from pathlib import Path
PD = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(PD.parent/'model')); OUT = PD/'data'/'model'
import physical_model as pm
scene, NC, SEED = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
BET = np.r_[np.arange(10, 86, 5.0), 89.0]; DT = 0.05; SIG = np.radians(20.0)
M = pm.Model(scene)
out = {'meta': dict(scene=scene, ncity=NC, seed=SEED, beta=BET.tolist(), beam_sigma_deg=20.0), 'res': []}
t_start = time.time()
for b in BET:
    t0 = pm.reference_time(b); s = pm.satellite(t0); los = (s-pm.UE)/np.linalg.norm(s-pm.UE)
    rng = np.random.default_rng(SEED)
    P = np.zeros(3); Pb = np.zeros(3); Nc = np.zeros(3); E1 = Wg = tg = ng = None
    dump = {k: [] for k in ('tau', 'nu', 'w', 'mech')} if (scene == 'urban' and b in (20., 80.)) else None
    for _ in range(NC):
        city = M.draw_city(rng); p = M.paths(t0, city)
        mech = p['mechanism']; w = p['amp']*p['prob']*p['weight']; g = mech == 3
        E1 = p['prob'][g][0]; Wg = p['amp'][g][0]; tg = p['tau'][g][0]
        # arrival-angle offset from boresight (LoS direction): ground 2*beta, walls/edges from the interaction point
        pts = p['points']; d = pts - pm.UE[:, None]; dn = np.linalg.norm(d, axis=0)
        ang = np.arccos(np.clip((los @ d)/np.maximum(dn, 1e-9), -1, 1)); ang[g] = 2*np.radians(b)
        G = np.exp(-ang**2/(2*SIG**2))
        for i in (1, 2):
            m = mech == i; P[i-1] += w[m].sum(); Pb[i-1] += (w[m]*G[m]).sum(); Nc[i-1] += p['prob'][m].sum()
        P[2] += Wg; Pb[2] += Wg*G[g][0]; Nc[2] += 1.0            # LoS state: ground weight W_g, one ground path
        if dump is not None:
            pp = M.paths(t0+DT, city); pmn = M.paths(t0-DT, city)
            _, i1, i2 = np.intersect1d(p['id'], pp['id'], assume_unique=True, return_indices=True)
            _, j1, j2 = np.intersect1d(p['id'], pmn['id'], assume_unique=True, return_indices=True)
            tp = np.full(len(w), np.nan); tm = np.full(len(w), np.nan); tp[i1] = pp['tau'][i2]; tm[j1] = pmn['tau'][j2]
            nu = -pm.FC*(tp-tm)/(2*DT); ok = np.isfinite(nu)
            wl = w.copy(); wl[g] = Wg
            dump['tau'].append(p['tau'][ok]); dump['nu'].append(nu[ok]); dump['w'].append(wl[ok]/NC); dump['mech'].append(mech[ok])
    P /= NC; Pb /= NC; Nc /= NC
    r = dict(beta=float(b), E1=float(E1), Wg=float(Wg), Pw=float(P[0]), Pe=float(P[1]), Pg=float(P[2]),
             Pw_beam=float(Pb[0]), Pe_beam=float(Pb[1]), Pg_beam=float(Pb[2]), n_wall=float(Nc[0]), n_edge=float(Nc[1]), n_ground=1.0,
             K_omni_dB=float(10*np.log10(1/P.sum())), K_beam_dB=float(10*np.log10(1/Pb.sum())))
    out['res'].append(r)
    if dump is not None:
        np.savez_compressed(OUT/f'joint_v2_{scene}_e{b:g}.npz', **{k: np.concatenate(v) for k, v in dump.items()})
    print(f"{scene:5s} {b:4.0f} Pg {P[2]:.4f} Pw {P[0]:.4f} Pe {P[1]:.2e} | contrib g/w/e {100*P[2]/P.sum():.1f}/{100*P[0]/P.sum():.1f}/{100*P[1]/P.sum():.1f} % | counts w {Nc[0]:.2f} e {Nc[1]:.2f} | K omni {r['K_omni_dB']:.2f} beam {r['K_beam_dB']:.2f} dB  [{time.time()-t_start:.0f}s]", flush=True)
json.dump(out, open(OUT/f'v2all_{scene}_n{NC}_s{SEED}.json', 'w'), indent=1)
