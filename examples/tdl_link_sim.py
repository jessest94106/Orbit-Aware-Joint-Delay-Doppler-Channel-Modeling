"""Example: driving an OFDM link simulation with the TDL channel of model/tdl.py.

What it does
  1. Draws one urban city (seed 237, the Fig. 10 scene) and starts the overhead pass at ELEV degrees.
  2. Every DT seconds takes a TDL snapshot h_k[n] (eq. tdl_cir) with tap spacing T_s = 1/30.72 MHz, i.e. the sample
     period of a 5G NR carrier with 30 kHz subcarrier spacing and a 1024-point FFT (20 MHz channel, 30.72 Msps).
  3. Pushes NSYM QPSK OFDM symbols per snapshot through the taps (time-domain convolution, cyclic prefix of 72
     samples = 2.3 us > the longest kept tap), adds white noise at the given Es/N0 measured on the direct path,
     equalises with the true per-subcarrier response H_m = sum_k h_k exp(-j 2 pi k m / N), and counts bit errors.
  4. Plots the tap powers, the per-subcarrier gain |H_m|^2 and the BER against time; prints a short summary.

Why snapshots are enough: after direct-path compensation every non-direct path carries a residual Doppler below
about 1 Hz (Sec. VI, Fig. 15), so the taps are constant over thousands of OFDM symbols; a link simulator can hold
h_k fixed for a slot and refresh it every 10-100 ms.  The direct path itself needs the usual LoS Doppler
pre-compensation, which is outside this channel model (Sec. III gives the closed-form LoS Doppler).

Usage: python3 examples/tdl_link_sim.py [ELEV=20] [DURATION_s=20] [DT_s=0.1] [EsN0_dB=10]
       (ELEV 80 with DURATION 40 shows a LoS blockage event at t = 28 s for this scene.)
Runtime: about 0.1 s per snapshot (ray tests against ~25 000 walls).
"""
import sys, time, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'model'))
from tdl import TDLChannel, rms_delay_spread

ELEV = float(sys.argv[1]) if len(sys.argv) > 1 else 20.
DURATION = float(sys.argv[2]) if len(sys.argv) > 2 else 20.
DT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
ESN0_DB = float(sys.argv[4]) if len(sys.argv) > 4 else 10.
NFFT, NCP, NSYM, NTAPS = 1024, 72, 20, 64
FS = 30.72e6; T_S = 1/FS
rng = np.random.default_rng(1)

# ---- 1. channel ----
ch = TDLChannel('urban', seed=237, T_s=T_S, n_taps=NTAPS)
t0 = ch.time_at_elevation(ELEV)
print(f'city: {len(ch.city["x"])} walls; pass from {ELEV:g} deg (t0 = {t0:.1f} s before zenith), {DURATION:g} s every {DT:g} s')

# ---- 2./3. OFDM link over each snapshot ----
def ofdm_ber(h, esn0_db, nsym=NSYM):
    """QPSK OFDM through taps h; noise set by Es/N0 of a unit-gain (direct-path-only) channel; ZF with true H."""
    bits = rng.integers(0, 2, (nsym, NFFT, 2))
    X = ((1-2*bits[..., 0]) + 1j*(1-2*bits[..., 1]))/np.sqrt(2)                       # unit-energy QPSK per subcarrier
    x = np.fft.ifft(X, axis=1)*np.sqrt(NFFT)                                           # unit average power per sample
    x_cp = np.concatenate([x[:, -NCP:], x], axis=1).ravel()                            # add CP, serialize
    y = np.convolve(x_cp, h)[:len(x_cp)]                                               # TDL convolution
    n0 = 10**(-esn0_db/10)                                                             # Es = 1 per sample on the direct path
    y += np.sqrt(n0/2)*(rng.standard_normal(len(y)) + 1j*rng.standard_normal(len(y)))
    Y = np.fft.fft(y.reshape(nsym, NFFT+NCP)[:, NCP:], axis=1)/np.sqrt(NFFT)          # strip CP, back to subcarriers
    H = np.fft.fft(h, NFFT)                                                            # per-subcarrier channel
    Xh = Y/H                                                                           # zero-forcing equaliser
    bh = np.stack([Xh.real < 0, Xh.imag < 0], axis=-1).astype(int)
    return float(np.mean(bh != bits)), H

ts = t0 + np.arange(0, DURATION+1e-9, DT); TAPS = []; BER = []; INFO = []; HF = []
t_wall = time.time()
for t in ts:
    h, info = ch.taps(t); ber, H = ofdm_ber(h, ESN0_DB)
    TAPS.append(h); BER.append(ber); INFO.append(info); HF.append(H)
TAPS = np.array(TAPS); HF = np.array(HF); BER = np.array(BER); trel = ts-t0
print(f'{len(ts)} snapshots in {time.time()-t_wall:.0f} s')

# ---- 4. summary and figure ----
los = np.array([i['los'] for i in INFO]); nw = np.array([i['n_wall'] for i in INFO]); ne = np.array([i['n_edge'] for i in INFO])
ds = np.array([rms_delay_spread(h, T_S) for h in TAPS])*1e9
pnd = np.array([i['power_nd'] for i in INFO]); dropped = np.array([i['power_dropped'] for i in INFO])
from scipy.special import erfc; awgn_ber = 0.5*erfc(np.sqrt(10**(ESN0_DB/10)))                # QPSK, Gray, per bit
print(f'elevation {INFO[0]["elevation_deg"]:.1f} -> {INFO[-1]["elevation_deg"]:.1f} deg; LoS clear in {los.mean()*100:.0f} % of snapshots; '
      f'wall paths {nw.min()}-{nw.max()}, roof-edge paths {ne.min()}-{ne.max()}; non-direct power {10*np.log10(pnd.min()):.1f} to {10*np.log10(pnd.max()):.1f} dB '
      f'(dropped beyond {NTAPS} taps: max {dropped.max():.1e}); RMS delay spread {ds.min():.1f}-{ds.max():.1f} ns')
print(f'QPSK OFDM at Es/N0 {ESN0_DB:g} dB: BER {BER.mean():.2e} (mean), {BER.max():.2e} (worst snapshot); AWGN reference {awgn_ber:.2e}')

fig, ax = plt.subplots(3, 1, figsize=(6.0, 6.8), sharex=True)
P = 10*np.log10(np.maximum(np.abs(TAPS)**2, 1e-6)).T
im = ax[0].imshow(P, origin='lower', aspect='auto', extent=[trel[0], trel[-1], -0.5, NTAPS-0.5], vmin=-40, vmax=0, cmap='viridis')
ax[0].set_ylabel(f'tap index k  ($T_s$ = {T_S*1e9:.1f} ns)'); ax[0].set_ylim(-0.5, 20.5); fig.colorbar(im, ax=ax[0], pad=0.01, label=r'$|h_k|^2$ [dB]')
ax[0].set_title(f'TDL taps, urban scene 237, from {ELEV:g}° (LoS clear {los.mean()*100:.0f} % of the time)', fontsize=9)
G = 10*np.log10(np.maximum(np.abs(np.fft.fftshift(HF, axes=1))**2, 1e-6)).T
f_mhz = (np.arange(NFFT)-NFFT/2)*FS/NFFT/1e6
im = ax[1].imshow(G, origin='lower', aspect='auto', extent=[trel[0], trel[-1], f_mhz[0], f_mhz[-1]], vmin=-20, vmax=5, cmap='viridis')
ax[1].set_ylabel('subcarrier frequency [MHz]'); fig.colorbar(im, ax=ax[1], pad=0.01, label=r'$|H_m|^2$ [dB]')
ax[2].semilogy(trel, np.maximum(BER, 1e-6), '-', color='#0072B2', lw=1.2, label=f'QPSK OFDM, ZF, Es/N0 {ESN0_DB:g} dB')
ax[2].axhline(max(awgn_ber, 1e-6), color='0.4', ls='--', lw=0.9, label='AWGN (direct path only)')
blk = ~los
if blk.any(): ax[2].fill_between(trel, 1e-6, 1, where=blk, color='#D55E00', alpha=0.15, lw=0, label='LoS blocked')
ax[2].set_ylim(1e-6, 1); ax[2].set_ylabel('BER'); ax[2].set_xlabel('time [s]'); ax[2].legend(fontsize=7.5, loc='upper right')
fig.colorbar(im, ax=ax[2], pad=0.01).ax.set_visible(False)                                  # keep the three panels the same width
for a in ax: a.grid(alpha=0.3)
fig.tight_layout(); out = ROOT/'examples'/f'tdl_link_sim_e{ELEV:g}.png'; fig.savefig(out, dpi=150); print('wrote', out)
