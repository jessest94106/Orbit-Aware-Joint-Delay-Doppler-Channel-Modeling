"""State-conditioned scene MC PDP (Section IV per-path city simulation, physical_model.py):
draw a Poisson city, build every wall / roof-edge / ground path, ray-test the interaction-to-GT segment
and the direct path against every other wall. The direct-path state of each scene is its own geometric
LoS event (visible[-1]); the ground bounce shares it. Histograms (33 ns bins) are accumulated separately
for LoS scenes and NLoS scenes and normalized by the non-direct power of the state.
Overhead pass, 2 GHz, omni.  Output data/scenemc/scenemc_state_SCENE_eELEV_nNSCENE_sSEED.json.
Usage: python3 paper_delivery/evaluate/scene_pdp_state.py SCENE ELEV NSCENE SEED   (paper: urban 6000-20000, rural 10000-42000 scenes per elevation)"""
import sys, time, json, numpy as np
from pathlib import Path
PD = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(PD.parent/'model')); OUT = PD/'data'/'scenemc'
import physical_model as pm
BIN = 33.; EDGES = np.arange(0, 6600+BIN, BIN)
def run(scene, elev, nscene, seed, chunk_log=2000):
    m = pm.Model(scene); t0 = pm.reference_time(elev); rng = np.random.default_rng(seed)
    H = {st: np.zeros((3, len(EDGES)-1)) for st in ('los', 'nlos')}; P = {st: np.zeros(3) for st in ('los', 'nlos')}
    n = {'los': 0, 'nlos': 0}; t = time.time()
    for k in range(nscene):
        city = m.draw_city(rng); p = m.paths(t0, city, geometric_blockage=True)
        st = 'los' if p['visible'][-1] else 'nlos'; n[st] += 1
        w = p['amp']*p['visible']*p['weight']; tau = p['tau']*1e9; mech = p['mechanism']
        for i in (1, 2, 3):
            s = mech == i
            H[st][i-1] += np.histogram(tau[s], bins=EDGES, weights=w[s])[0]; P[st][i-1] += w[s].sum()
        if (k+1) % chunk_log == 0: print(f'{scene} {elev:g} {k+1}/{nscene} n_los {n["los"]} n_nlos {n["nlos"]} {time.time()-t:.0f}s', flush=True)
    out = dict(scene=scene, elev=elev, nscene=nscene, seed=seed, edges=EDGES.tolist(), n=n)
    for st in ('los', 'nlos'):
        if n[st]:
            Pn = P[st].sum()/n[st]
            out[st] = dict(pdp=(H[st].sum(0)/n[st]/Pn).tolist(), P_wall=P[st][0]/n[st], P_edge=P[st][1]/n[st], P_ground=P[st][2]/n[st], P_nd=Pn,
                           pdp_wall=(H[st][0]/n[st]/Pn).tolist(), pdp_edge=(H[st][1]/n[st]/Pn).tolist())
    out['P_los'] = n['los']/nscene
    return out
if __name__ == '__main__':
    scene, elev, nscene, seed = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    t = time.time(); out = run(scene, elev, nscene, seed); out['seconds'] = time.time()-t
    json.dump(out, open(OUT/f'scenemc_state_{scene}_e{elev:g}_n{nscene}_s{seed}.json', 'w'))
    print(f"{scene} {elev:g} n={nscene} {out['seconds']:.0f}s P_los {out['P_los']:.4f} n_nlos {out['n']['nlos']} | LoS: wall {out['los']['P_wall']:.5f} edge {out['los']['P_edge']:.2e} ground {out['los']['P_ground']:.4f}"
          + (f" | NLoS: wall {out['nlos']['P_wall']:.5f} edge {out['nlos']['P_edge']:.2e}" if 'nlos' in out else ''), flush=True)
