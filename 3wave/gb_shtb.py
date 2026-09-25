import sys, numpy as np
sys.path.insert(0, '.')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import cases
import config
from wave_separation import separate, separate_time_domain

def gb(sg1, sg2, fs, x1, x2, xt, c0, alpha_fac=0.05):
    """Verbatim port of separate_waves_nodisp (2waves_deconvolution_GB.md)."""
    N = len(sg1); Nf = int(2 ** np.ceil(np.log2(3 * N)))
    f = fs * np.arange(Nf) / Nf; w = 2 * np.pi * f
    w[Nf // 2 + 1:] -= 2 * np.pi * fs
    k = w / c0; a = alpha_fac / abs(x2 - x1); g = a + 1j * k
    E1 = np.fft.fft(sg1, Nf); E2 = np.fft.fft(sg2, Nf); dx = x2 - x1
    den = np.exp(g * dx) - np.exp(-g * dx)
    A = (E1 * np.exp(g * dx) - E2) / den; B = (E2 - E1 * np.exp(-g * dx)) / den
    d = xt - x1
    return (np.real(np.fft.ifft(A * np.exp(-g * d)))[:N],
            np.real(np.fft.ifft(B * np.exp(g * d)))[:N])

# usage: python3 gb_shtb.py [CASE_DIR] [OUT.png]   (default cases/analyses/SHTB_PC)
CFG = config.load(sys.argv[1] if len(sys.argv) > 1 else 'cases/analyses/SHTB_PC')
ID = cases.identification(CFG)
d = cases.record(CFG); t, dt = d['t'], d['dt']; eta = d['eta']
thr = 0.05
F = {}
for b in ('in', 'out'):
    sig = list(d[f'eps_{b}']); c0 = float(ID[f'c_{b}']); x = np.asarray(ID[f'x_{b}'], float)
    att = (ID[f'alpha_f_{b}'], ID[f'alpha_{b}']); disp = (ID[f'dispersion_f_{b}'], ID[f'dispersion_{b}'])
    r = {}
    for fac in (0.05, 0.005):
        # GB's coordinate: SG1 at x1, SG2 at x2, target at the face, x = 0
        fw, bw = gb(sig[0], sig[1], 1 / dt, x[0], x[1], 0.0, c0, fac); r[f'GB a={fac}/dx'] = fw + bw
    p, m = separate(t, sig, x, c0, eta); r['separate, lossless'] = p + m
    p, m = separate(t, sig, x, c0, eta, dispersion=disp, attenuation=att); r['separate, att+disp (project default)'] = p + m
    p, m = separate_time_domain(t, sig, x, c0, arrival_frac=thr); r['time domain'] = p + m
    F[b] = r
    print(f'\n--- {b} bar, c0={c0:.1f} mm/ms, x={x}, dx={abs(x[1]-x[0]):.1f} mm, '
          f'GB alpha={0.05/abs(x[1]-x[0]):.2e}/mm vs eta/c0={eta/c0:.2e}/mm')
    ref = r['separate, lossless']; full = r['separate, att+disp (project default)']
    pk = np.abs(ref).max()
    print(f'{"method":40s} {"peak |F| [N]":>12} {"rms vs lossless":>16} {"rms vs default":>15} {"F at end [N]":>13}')
    for k, v in r.items():
        print(f'{k:40s} {np.abs(v).max():12.1f} {np.sqrt(np.mean((v-ref)**2))/pk:16.3%} '
              f'{np.sqrt(np.mean((v-full)**2))/pk:15.3%} {np.mean(v[-200:]):13.1f}')

print('\n--- force equilibrium, rms |F_in - F_out| / peak F_out, over the record')
for k in F['in']:
    e = F['in'][k] - F['out'][k]
    print(f'{k:40s} {np.sqrt(np.mean(e**2))/np.abs(F["out"][k]).max():8.3f}')

fig, ax = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
tu = t * 1e3
for i, b in enumerate(('in', 'out')):
    for k, v in F[b].items():
        ax[i].plot(tu, v, lw=1.0 if 'GB' in k else 1.4, ls='--' if 'GB a=0.005' in k else '-', label=k)
    ax[i].set_ylabel(f'F at {b}-bar face [N]'); ax[i].legend(fontsize=8); ax[i].grid(alpha=.3)
for k in ('GB a=0.05/dx', 'separate, lossless', 'separate, att+disp (project default)'):
    ax[2].plot(tu, F['in'][k] - F['out'][k], label=k)
ax[2].set_ylabel('F_in - F_out [N]'); ax[2].set_xlabel('t [us]'); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
fig.suptitle('SHTB_PC: GB separate_waves_nodisp vs this project')
fig.tight_layout(); fig.savefig(sys.argv[2] if len(sys.argv) > 2 else cases.output(CFG, 'gb_shtb.png'), dpi=110)
