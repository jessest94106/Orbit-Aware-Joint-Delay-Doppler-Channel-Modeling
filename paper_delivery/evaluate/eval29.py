"""Corrected numerical evaluation of eq. (29) ("model v2"), replacing gt_segment model.py:
uniform PPP walls (physical_model.draw_city), wall image at the ANTENNA height, exact roof-edge stationary point
(Newton), analytic finite-segment visibility probability (the GT-segment rule of Sec. IV / App. B), ground bounce
gated by the LoS event, omni, 2 GHz, overhead pass.  Doppler by finite difference of each path's delay (dt = 50 ms).
States: LoS = ground W_g + walls + edges; NLoS = walls + edges; per-state PDP (33 ns), DS, sigma_nu, T_c.
Outputs data/model/eval29_SCENE_nNCITY_sSEED.json (per-elevation DS / sigma_nu / T_c per state, PDPs at PDP_E).
Usage: python3 paper_delivery/evaluate/eval29.py SCENE NCITY SEED [ELEV ...]   (default elevations 10:5:80; PDPs are stored at 10/20/40/80 deg)
Paper runs: urban 6000 3 / rural 6000 4 (all elevations, ~35 / 55 min on one core);
            urban 30000 11 / rural 30000 12 (10 40 80, PDP only); urban 30000 21 / rural 30000 22 (20, PDP only)."""
import sys, json, time, numpy as np
from pathlib import Path
PD = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(PD.parent/'model')); OUT = PD/'data'/'model'
import physical_model as pm
scene, NC, SEED = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
BET = np.array([float(a) for a in sys.argv[4:]]) if len(sys.argv) > 4 else np.arange(10, 81, 5.0); PDP_E = (10., 20., 40., 80.); BIN = 33e-9; NB = 200; DT = 0.05
M = pm.Model(scene); rng = np.random.default_rng(SEED)
def mom(tau, nu, w):
    P = w.sum(); tb = (w*tau).sum()/P; nb = (w*nu).sum()/P
    return float(np.sqrt(max((w*tau*tau).sum()/P-tb*tb, 0))), float(np.sqrt(max((w*nu*nu).sum()/P-nb*nb, 0)))
out = {'meta': dict(scene=scene, ncity=NC, seed=SEED, beta=BET.tolist(), pdp_elev=PDP_E, bin_ns=33, evaluator='eval29 (physical_model, analytic visibility)'), 'res': [], 'pdp': {}}
t_start = time.time()
for b in BET:
    t0 = pm.reference_time(b)
    acc = {st: dict(T=[], N=[], W=[], mech=[]) for st in ('los', 'nlos')}
    E1 = None; Wg = None; tg = None; ng = None
    rng = np.random.default_rng(SEED)                       # same cities at every elevation, drawn on the fly (memory)
    for _ in range(NC):
        city = M.draw_city(rng); p = M.paths(t0, city); pp = M.paths(t0+DT, city); pmn = M.paths(t0-DT, city)
        _, i1, i2 = np.intersect1d(p['id'], pp['id'], assume_unique=True, return_indices=True)
        _, j1, j2 = np.intersect1d(p['id'], pmn['id'], assume_unique=True, return_indices=True)
        nu = np.full(len(p['id']), np.nan); tp = np.full(len(p['id']), np.nan); tm = np.full(len(p['id']), np.nan)
        tp[i1] = pp['tau'][i2]; tm[j1] = pmn['tau'][j2]; nu = -pm.FC*(tp-tm)/(2*DT)
        ok = np.isfinite(nu)                                   # paths that exist at all three instants
        mech = p['mechanism']; w = p['amp']*p['prob']*p['weight']
        g = mech == 3; E1 = p['prob'][g][0]; Wg = p['amp'][g][0]; tg = p['tau'][g][0]; ng = nu[g][0]
        ne = ok & (mech != 3)
        acc['nlos']['T'].append(p['tau'][ne]); acc['nlos']['N'].append(nu[ne]); acc['nlos']['W'].append(w[ne]); acc['nlos']['mech'].append(mech[ne])
    for st in ('los', 'nlos'):
        for k in ('T', 'N', 'W', 'mech'): acc[st][k] = np.concatenate(acc['nlos'][k]) if acc['nlos'][k] else np.zeros(0)
    T, N, W, MECH = acc['nlos']['T'], acc['nlos']['N'], acc['nlos']['W']/NC, acc['nlos']['mech']   # per-city expected
    Pw, Pe = W[MECH == 1].sum(), W[MECH == 2].sum()
    r = dict(beta=float(b), E1=float(E1), Wg=float(Wg), Pw=float(Pw), Pe=float(Pe), Pnlos_los=float(Wg+Pw+Pe), Pnlos_nlos=float(Pw+Pe), nu_g_Hz=float(ng), tau_g_ns=float(tg*1e9))
    # LoS state: ground atom (Wg) + walls + edges; direct path weight 1 at (0,0)
    TL = np.r_[tg, T]; NL = np.r_[ng, N]; WL = np.r_[Wg, W]
    r['ds_los_ns'] = mom(np.r_[0., TL], np.r_[0., NL], np.r_[1., WL])[0]*1e9; r['dsp_los_Hz'] = mom(TL, NL, WL)[1]; r['Tc_los_s'] = 1/(2*np.pi*r['dsp_los_Hz'])
    r['ds_nlos_ns'] = mom(T, N, W)[0]*1e9; r['dsp_nlos_Hz'] = mom(T, N, W)[1]; r['Tc_nlos_s'] = 1/(2*np.pi*r['dsp_nlos_Hz'])
    TU = np.r_[tg, T]; WU = np.r_[E1*Wg, W]
    r['ds_uncond_ns'] = mom(np.r_[0., TU], np.r_[0., NL], np.r_[E1, WU])[0]*1e9; r['dsp_uncond_Hz'] = mom(TU, NL, WU)[1]
    r['dsp_wall_Hz'] = mom(T[MECH == 1], N[MECH == 1], W[MECH == 1])[1]; r['dsp_edge_Hz'] = mom(T[MECH == 2], N[MECH == 2], W[MECH == 2])[1] if Pe > 0 else None
    r['ds_wall_ns'] = mom(T[MECH == 1], N[MECH == 1], W[MECH == 1])[0]*1e9
    out['res'].append(r)
    if b in PDP_E:
        edges = np.arange(0, NB+1)*BIN
        for st, TT, WW in (('los', TL, WL), ('nlos', T, W), ('uncond', TU, WU)):
            h = np.histogram(TT, bins=edges, weights=WW/WW.sum())[0]
            out['pdp'][f'{scene}_{st}_{b:g}'] = dict(centers_ns=((edges[:-1]+0.5*BIN)*1e9).tolist(), p=h.tolist(), tail=float(WW[TT > edges[-1]].sum()/WW.sum()))
    print(f"{scene:5s} {b:4.0f} E1 {E1:.3f} Wg {Wg:.4f} Pw {Pw:.4f} Pe {Pe:.2e} | DS los {r['ds_los_ns']:6.1f} nlos {r['ds_nlos_ns']:6.1f} ns | sig_nu los {r['dsp_los_Hz']:.4f} nlos {r['dsp_nlos_Hz']:.4f} Hz  [{time.time()-t_start:.0f}s]", flush=True)
json.dump(out, open(OUT/f'eval29_{scene}_n{NC}_s{SEED}.json', 'w'))
