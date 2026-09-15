"""Tapped-delay-line (TDL) channel of Sec. IV-F built from the per-path model on one drawn city.

    h_k[n] = 1_0(t_n) delta_{k0}
           + sum over visible non-direct paths i with floor(tau_i(t_n)/T_s) = k of
             sqrt(W_i(t_n)) exp(j phi_i) exp(-j 2 pi f_c tau_i(t_n))                     (eq. tdl_cir)

* tau_i is the excess delay of path i relative to the direct path, so the LoS Doppler is already compensated: tap 0
  holds the (unit, zero-phase) direct path whenever it is geometrically clear, and every other path carries the
  residual phase 2 pi f_c tau_i(t), i.e. its residual Doppler nu_i = -f_c dtau_i/dt (|nu_i| below ~1 Hz).
* phi_i is one random initial phase per path (uniform, fixed for the life of the path, keyed by the path id).
* Visibility is ray-traced in the drawn city (Model.paths(..., geometric_blockage=True)): the direct path against
  every wall, each wall/roof-edge path from its interaction point to the terminal; the ground reflection shares the
  direct-path indicator (eq. existence).
* Powers are relative to the free-space direct path (direct path = 1); apply your own path loss / noise scaling.

Typical use (see examples/tdl_link_sim.py):
    ch = TDLChannel('urban', seed=237, T_s=1/30.72e6, n_taps=64)
    t0 = ch.time_at_elevation(20.)                 # satellite at 20 deg, ascending
    h, info = ch.taps(t0 + 0.5)                    # complex taps at t0 + 0.5 s
    H = np.fft.fft(h, 1024)                        # channel per subcarrier of a 1024-point OFDM system at 1/T_s Hz
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import physical_model as pm

class TDLChannel:
    def __init__(self, scene='urban', seed=0, T_s=1/30.72e6, n_taps=64, radius=2000., phase_seed=None):
        """scene 'urban' | 'rural'; seed draws the city; T_s tap spacing [s]; n_taps taps kept (paths beyond are dropped
        and reported in info['power_dropped']); radius of the drawn city [m]."""
        self.model = pm.Model(scene, radius=radius)
        self.city = self.model.draw_city(np.random.default_rng(seed))
        self.T_s, self.n_taps, self.fc = float(T_s), int(n_taps), pm.FC
        self._phase_rng = np.random.default_rng(seed if phase_seed is None else phase_seed)
        self._phase = {}                                    # path id -> initial phase phi_i

    @staticmethod
    def time_at_elevation(elev_deg):
        """Time [s] (zenith at t = 0, ascending side negative) at which the satellite is at elev_deg."""
        return pm.reference_time(float(elev_deg))

    @staticmethod
    def elevation(t):
        return float(np.degrees(pm.elevation(t)))

    def phase(self, path_id):
        if path_id not in self._phase: self._phase[path_id] = self._phase_rng.uniform(0, 2*np.pi)
        return self._phase[path_id]

    def paths(self, t):
        """Visible paths at time t: dict with mechanism (1 wall, 2 roof edge, 3 ground), tau [s], power, id, and los."""
        p = self.model.paths(t, self.city, geometric_blockage=True)
        los = bool(p['visible'][-1]); keep = p['visible'].copy(); keep[-1] = los     # ground bounce gated by the LoS event
        return dict(los=los, mechanism=p['mechanism'][keep], tau=p['tau'][keep], power=p['amp'][keep], id=p['id'][keep])

    def taps(self, t):
        """Complex taps h[0..n_taps-1] at time t (eq. tdl_cir) and an info dict."""
        q = self.paths(t); h = np.zeros(self.n_taps, complex)
        if q['los']: h[0] += 1.0
        k = np.floor(q['tau']/self.T_s).astype(int); inside = k < self.n_taps
        for kk, pid, tau, w in zip(k[inside], q['id'][inside], q['tau'][inside], q['power'][inside]):
            h[kk] += np.sqrt(w)*np.exp(1j*self.phase(int(pid)))*np.exp(-2j*np.pi*self.fc*tau)
        info = dict(t=float(t), elevation_deg=self.elevation(t), los=q['los'],
                    n_wall=int(np.sum(q['mechanism'] == 1)), n_edge=int(np.sum(q['mechanism'] == 2)),
                    ground=bool(np.any(q['mechanism'] == 3)), power_nd=float(q['power'].sum()),
                    power_dropped=float(q['power'][~inside].sum()))
        return h, info

    def sequence(self, t_start, duration, dt):
        """Taps on the grid t_start + n dt, n dt <= duration: array (N, n_taps) and the list of info dicts."""
        ts = t_start + np.arange(0, duration+1e-9, dt); H = []; I = []
        for t in ts:
            h, i = self.taps(t); H.append(h); I.append(i)
        return np.array(H), I

def rms_delay_spread(h, T_s):
    """RMS delay spread [s] of a tap vector (direct path included)."""
    p = np.abs(h)**2; k = np.arange(len(h))*T_s; P = p.sum()
    if P == 0: return 0.0
    m = (p*k).sum()/P; return float(np.sqrt(max((p*k*k).sum()/P-m*m, 0)))
