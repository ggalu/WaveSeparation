# Wave separation in Python

A Python port of the three-gauge wave separation in `Prog_Treat/a05.m`
(`wave_separation3`), together with the reduction from separated waves to
specimen stress/strain, and a validation against a 1D Hopkinson bar simulation.

## Quick start

```bash
python3 drive.py             # ~1 s  -> dump.npz
python3 reduce_specimen.py   # ~2 s  -> specimen_reconstruction.png
```

`drive.py` runs the simulation and records the gauge signals;
`reduce_specimen.py` does the actual wave reconstruction. You only re-run
`drive.py` when you change a simulation parameter — the reconstruction reads
`dump.npz`, so iterating on the analysis is a 2-second loop.

There is a second simulator, a Split Hopkinson **tension** bar driven by a
striker. It writes the same filename, so run one or the other:

```bash
python3 drive_tension.py     # SHTB instead of the compression bar
```

Its data is TENSION POSITIVE, and the dump says so — `reduce_specimen.py` reads
the sign convention rather than being told. Unlike the compression model it
produces genuine incident/reflected **overlap** at the input-bar gauge, which is
the case multi-gauge separation exists for.

## Everything is configured in one file

`config.toml` holds both cases — materials, geometry, gauge locations, numerics
and the analysis `eta`. The drivers and the analysis scripts read it; nothing is
hardcoded in the Python any more. To move a gauge or change the striker, edit
`config.toml` and re-run the driver.

The two cases are independent and have their own gauge lists:

```toml
[tension.striker]        # POM tube
E = 3.0
inner_diameter = 16.1
outer_diameter = 40.0
length = 800.0

[tension]
gauges = [130.0, 530.0, 1177.0]   # mm from the interface plane
```

## What gets recorded

The simulators used to keep strain and force for **every element at every
timestep** — for the tension case a 6030 × 36861 pair of arrays, 3.6 GB in
memory and 1.8 GB on disk, to produce six gauge signals. They now record only
what is consumed:

| Recorded | Why |
|---|---|
| strain at each gauge element | the input to the separation |
| force in the two elements bounding the specimen | ground truth for `sep_test.py` |
| mean specimen strain and force | ground truth for the reduction; accumulated per step, not stored per element |

That is ~10 numbers per timestep instead of ~12000: **`dump.npz` is 2.9 MB and
peak memory is 33 MB.** Set `record_full_field = true` in `config.toml` to get
the old every-element arrays back (as `eps_full` / `force_full`) when you need
them for an animation or a Lagrange diagram.

Two optional extras, both reading the same dump:

```bash
python3 plot_forces.py       # raw gauge signals vs specimen force
python3 sep_test.py          # separation accuracy vs ground truth, eta sweep
```

`reduce_specimen.py` prints the validation and writes `specimen_reconstructed.dat`
(time, stress, strain, strain rate — compression positive) plus a three-panel
figure.

To remove every generated file and start clean:

```bash
./clean.sh          # remove them
./clean.sh -n       # dry run: list what would go, delete nothing
```

## Theory

### The problem

A strain gauge on a Hopkinson bar measures one number: the total strain passing
that point. But two waves pass it — one heading toward the specimen, one heading
away — and what you actually want is the force and velocity at the **bar/specimen
interface**, where no gauge can be placed.

The classical way out is to put the gauge far enough from the specimen that the
incident and reflected pulses arrive at *separate times*, so each can be read off
the record in isolation and shifted to the interface by hand. That works only
while the pulses stay apart. Make the pulse longer, the specimen softer, or the
strain larger — exactly what you do to reach high strains — and the two waves
**overlap** at the gauge. One measurement, two unknowns, and no way to tell them
apart.

Wave separation is the way out: measure at **several points along the bar**. Each
gauge sees the same two waves but with different phase, because they have
travelled different distances to get there. That phase difference is what makes
the two recoverable.

### The model, per gauge

Each bar is treated in its own local coordinate $x$, measured from the specimen
interface and positive going *into* the bar. Strain at a gauge at distance $x_k$
is the superposition of the two travelling waves evaluated at that point:

```math
\varepsilon_k(\omega) \;=\; \underbrace{P(\omega)\,e^{-i\xi x_k}}_{\text{away from specimen}}
             \;+\; \underbrace{M(\omega)\,e^{+i\xi x_k}}_{\text{toward specimen}}
```

where

```math
\xi \;=\; \frac{\omega - i\eta}{c_p}
```

is the complex wavenumber. Its imaginary part is where the exponential window
$e^{-\eta t}$, applied before the FFT, enters: it shifts the transform off the
real frequency axis.

The unknowns $P$ and $M$ are the two waves **at $x = 0$**, the interface. Every
gauge sees the same two unknowns — only the phase factors differ.

### Where the Laplace transform is hiding

The method is described as Laplace-domain, yet the code calls nothing but
`np.fft.rfft` and `np.fft.irfft`. There is no contradiction: **multiplying by
$e^{-\eta t}$ before a Fourier transform *is* a Laplace transform.** Writing the
one-sided Laplace transform with $s = \eta + i\omega$,

```math
\mathcal{L}\{f\}(s) = \int_0^\infty \! f(t)\,e^{-st}\,dt
                    = \int_0^\infty \! \underbrace{f(t)\,e^{-\eta t}}_{\text{the window}}
                      e^{-i\omega t}\,dt
                    = \mathcal{F}\bigl\{f\,e^{-\eta t}\bigr\}(\omega)
```

So the two lines in `separate`

```python
win = np.exp(-eta * tau)
E   = np.fft.rfft(s * win, n_fft)
```

evaluate the Laplace transform along the **vertical line $\mathrm{Re}(s) = \eta$**
in the complex $s$-plane — the Bromwich contour. The FFT supplies the $i\omega$
sweep along that line; the window supplies the offset $\eta$ from the imaginary
axis. Verified against a case with a known transform: for $f = e^{-at}$ the
windowed FFT reproduces $1/(s+a)$ to 2e-04 (rectangle-rule discretisation error).

**$\xi$ is that same $s$ in disguise.** A wave travelling at $c_p$ carries the
Laplace-domain propagator $e^{-s x/c_p}$, and

```math
e^{-s x / c_p} = e^{-(\eta + i\omega)x/c_p} = e^{-i\xi x},
\qquad
\xi = \frac{\omega - i\eta}{c_p} = \frac{s}{i\,c_p}
```

which is exactly the wavenumber used above. The inverse is the matching pair:
`irfft` followed by multiplication by $e^{+\eta t}$ is a discretised Bromwich
integral — the Fourier-series (Dubner–Abate / Durbin) method of numerical Laplace
inversion.

Setting $\eta = 0$ collapses $s$ onto the imaginary axis and the whole thing
degenerates to an ordinary Fourier transform. That is permitted arithmetically
and fatal numerically: on the imaginary axis the system determinant vanishes at
DC and at every half-wavelength-commensurate frequency. **Stepping off the real
frequency axis is the entire purpose of the window** — the regularisation and the
Laplace transform are the same act.

It is not free. The inversion multiplies by $e^{+\eta t}$, which amplifies
whatever sits at the end of the record, so the window is only invertible while
$\eta T$ stays modest:

| $\eta T$ | $e^{+\eta T}$ | round-trip error |
|---|---|---|
| 0 | 1 | 3e-16 |
| 5 | 1.5e+02 | 2e-14 |
| 10 | 2.2e+04 | 4e-12 |
| 30 | 1.1e+13 | 1e-03 |
| 70 | 2.5e+30 | 2e+14 |

`separate` refuses above $\eta T = 700$, where the float overflows outright, but
as the table shows it is already useless by 30. This is the same ceiling that
makes the large-$\eta$ rows of `sep_test.py` blow up.

### Generalising to any number of gauges

Write $a_k = e^{-i\xi x_k}$ and $b_k = e^{+i\xi x_k}$ for the two phase factors
at gauge $k$, so the model of the previous section reads
$\varepsilon_k = P a_k + M b_k$.

#### Two gauges: an exact solve

With exactly two gauges there are two equations and two unknowns, and no
approximation is involved:

```math
\begin{bmatrix} a_1 & b_1 \\[2pt] a_2 & b_2 \end{bmatrix}
\begin{bmatrix} P \\[2pt] M \end{bmatrix}
=
\begin{bmatrix} \varepsilon_1 \\[2pt] \varepsilon_2 \end{bmatrix},
\qquad
\mathbf{A}\mathbf{z} = \boldsymbol{\varepsilon}
```

The determinant collapses to a single sine. With $D = x_2 - x_1$ the gauge
spacing,

```math
\det\mathbf{A} = a_1b_2 - b_1a_2
       = e^{+i\xi D} - e^{-i\xi D}
       = 2i\,\sin(\xi D)
```

so Cramer's rule gives the separated waves outright:

```math
P = \frac{\varepsilon_1 e^{+i\xi x_2} - \varepsilon_2 e^{+i\xi x_1}}{2i\,\sin(\xi D)},
\qquad
M = \frac{\varepsilon_2 e^{-i\xi x_1} - \varepsilon_1 e^{-i\xi x_2}}{2i\,\sin(\xi D)}
```

**Everything about the method is already visible here.** The solution exists
unless $\sin(\xi D) = 0$, and only the spacing $D$ appears — not where the pair
sits on the bar. Splitting $\xi D$ into its real and imaginary parts,

```math
\bigl|\sin(\xi D)\bigr|^2
  = \sin^2\!\left(\frac{\omega D}{c}\right) + \sinh^2\!\left(\frac{\eta D}{c}\right)
```

With $\eta = 0$ the second term vanishes and the first is zero whenever
$\omega D/c = n\pi$, i.e. whenever the spacing is a whole number of
half-wavelengths: the two gauges then see the same phase and cannot tell the
waves apart. The $\sinh$ term — which exists only because $\eta > 0$ — is what
keeps the denominator away from zero.

#### More than two gauges: least squares

With $K > 2$ gauges, $\mathbf{A}$ is $K \times 2$ and
$\mathbf{A}\mathbf{z} = \boldsymbol{\varepsilon}$ is overdetermined: measurement
noise and model error mean no $(P, M)$ satisfies all $K$ equations at once. Take
the pair that comes closest in the least-squares sense,

```math
\min_{P,\,M}\; J(P,M), \qquad
J = \bigl\|\boldsymbol{\varepsilon} - \mathbf{A}\mathbf{z}\bigr\|^2
  = \sum_{k=1}^{K} \bigl| \varepsilon_k - P\,a_k - M\,b_k \bigr|^2
```

$J$ is real and quadratic, so its minimum is where the derivatives with respect
to $\bar{P}$ and $\bar{M}$ both vanish:

```math
\frac{\partial J}{\partial \bar{P}}
 = -\sum_k \bar{a}_k\bigl(\varepsilon_k - P a_k - M b_k\bigr) = 0,
\qquad
\frac{\partial J}{\partial \bar{M}}
 = -\sum_k \bar{b}_k\bigl(\varepsilon_k - P a_k - M b_k\bigr) = 0
```

Rearranging each into unknowns-on-the-left form, and writing
$\langle \mathbf{u},\mathbf{v}\rangle = \sum_k \bar{u}_k v_k$,

```math
P\,\langle \mathbf{a},\mathbf{a}\rangle + M\,\langle \mathbf{a},\mathbf{b}\rangle
  = \langle \mathbf{a},\boldsymbol{\varepsilon}\rangle,
\qquad
P\,\langle \mathbf{b},\mathbf{a}\rangle + M\,\langle \mathbf{b},\mathbf{b}\rangle
  = \langle \mathbf{b},\boldsymbol{\varepsilon}\rangle
```

which is $\mathbf{A}^{H}\mathbf{A}\,\mathbf{z} = \mathbf{A}^{H}\boldsymbol{\varepsilon}$ —
**the $K$ equations have collapsed back into a $2 \times 2$ system:**

```math
\begin{bmatrix}
\langle \mathbf{a},\mathbf{a}\rangle & \langle \mathbf{a},\mathbf{b}\rangle \\[2pt]
\langle \mathbf{b},\mathbf{a}\rangle & \langle \mathbf{b},\mathbf{b}\rangle
\end{bmatrix}
\begin{bmatrix} P \\[2pt] M \end{bmatrix}
=
\begin{bmatrix}
\langle \mathbf{a},\boldsymbol{\varepsilon}\rangle \\[2pt]
\langle \mathbf{b},\boldsymbol{\varepsilon}\rangle
\end{bmatrix}
```

Every entry is a **sum over gauges**, and that is the whole generalisation:

| Entry | Value | In `separate` |
|---|---|---|
| $\langle \mathbf{a},\mathbf{a}\rangle$ | $\sum_k e^{-i(\xi - \bar\xi)x_k}$ | `h1` |
| $\langle \mathbf{b},\mathbf{b}\rangle$ | $\sum_k e^{+i(\xi - \bar\xi)x_k}$ | `h2` |
| $\langle \mathbf{a},\mathbf{b}\rangle$ | $\sum_k e^{+i(\xi + \bar\xi)x_k}$ | `g` |
| $\langle \mathbf{a},\boldsymbol{\varepsilon}\rangle$ | $\sum_k \varepsilon_k\,e^{+i\bar\xi x_k}$ | `E1` |
| $\langle \mathbf{b},\boldsymbol{\varepsilon}\rangle$ | $\sum_k \varepsilon_k\,e^{-i\bar\xi x_k}$ | `E2` |

The gauge count is never a dimension — it is only the number of terms in five
sums. The system solved is always $2 \times 2$, whether there are 2 gauges or 20.
Because $\xi - \bar\xi = -2i\eta/c_p$ and $\xi + \bar\xi = 2\omega/c_p$, these
reduce to

```math
h_1 = \sum_k e^{-2\eta x_k/c} \;>\; 0,
\qquad
h_2 = \sum_k e^{+2\eta x_k/c} \;>\; 0,
\qquad
g   = \sum_k e^{+2i\omega x_k/c}
```

so $h_1$ and $h_2$ are real and positive while $g$ is a sum of pure phases, and
Cramer's rule finishes it with

```math
\det \;=\; h_1 h_2 - |g|^2
```

#### The two routes are the same route

For $K = 2$ the least-squares machinery does **not** give a different answer from
the exact solve above. $\mathbf{A}$ is then square and invertible, so

```math
\mathbf{z} = \bigl(\mathbf{A}^{H}\mathbf{A}\bigr)^{-1}\mathbf{A}^{H}\boldsymbol{\varepsilon}
           = \mathbf{A}^{-1}\bigl(\mathbf{A}^{H}\bigr)^{-1}\mathbf{A}^{H}\boldsymbol{\varepsilon}
           = \mathbf{A}^{-1}\boldsymbol{\varepsilon}
```

The residual $J$ is zero and the fit is an interpolation. Numerically, the
explicit Cramer expressions above and `separate` agree to $6 \times 10^{-14}$
relative on the shipped 2-gauge dump.

The determinants match too, which is worth noting because the next subsection
leans on it: for a square $\mathbf{A}$,
$\det(\mathbf{A}^{H}\mathbf{A}) = |\det\mathbf{A}|^2$, so

```math
h_1 h_2 - |g|^2 \;=\; \bigl|2i\sin(\xi D)\bigr|^2 \;=\; 4\bigl|\sin(\xi D)\bigr|^2
```

**So there is one code path, not two.** With two gauges it interpolates exactly;
with more it becomes a genuine fit that averages the redundancy; and the only
thing that changes between them is how many terms are in the five sums. This is
why a 2+2 layout needs no special-casing, and why `separate` accepts any
$K \ge 2$ without a branch.

### What the determinant tells you

$\det$ is the Gram determinant of $\mathbf{a}$ and $\mathbf{b}$, so it vanishes
exactly when the two are parallel — when the gauge array cannot tell an outgoing
wave from an incoming one. For **two** gauges with spacing $D$ it follows from
$4|\sin(\xi D)|^2$ above, via the same real/imaginary split and a double-angle
identity:

```math
\det \;=\; 2\left[\, \cosh\!\left(\frac{2\eta D}{c}\right)
                   - \cos\!\left(\frac{2\omega D}{c}\right) \right]
```

Both failure modes are visible in it:

- the **cosine** term dips whenever $2\omega D/c = 2\pi n$, i.e. every $c/2D$ in
  frequency — the spacing is a whole number of half-wavelengths and the phase
  factors come back into step. For the shipped $D = 400\ \text{mm}$ that is every
  **6.31 kHz**;
- the **cosh** term is what stops those dips reaching zero. At $\eta = 1.0$ /ms
  the floor is $2[\cosh(2\eta D/c) - 1] = 2.5 \times 10^{-2}$ against a typical
  value of $2.0$, so roughly 1 %: noise there is amplified about 80x, but nothing
  is divided by zero. At $\eta = 0$ the cosh term is exactly 1, the dips touch
  zero, and the system is singular at DC as well.

That is why `eta` is mandatory, and why `conditioning()` exists — use it to audit
a layout before committing to it. See [Choosing eta](#choosing-eta) for how to
pick the value.

### From the separated waves to the force at the specimen

The two bars are solved **completely independently**; they share nothing until
the very last step. For a 2+2 layout the chain runs:

1. **Per bar** (`separate`): two gauge signals → windowed FFT → for each
   frequency bin, solve that $2 \times 2$ system → $P(\omega), M(\omega)$ →
   inverse FFT → undo the window with $e^{+\eta t}$ → $\varepsilon_+(t)$ and
   $\varepsilon_-(t)$, both now at $x = 0$.

2. **Interface state** (`bar_interface`):

   ```math
   F = EA\,(\varepsilon_+ + \varepsilon_-),
   \qquad
   v_\text{local} = c_0\,(\varepsilon_- - \varepsilon_+)
   ```

   Force *adds* the two waves because strain superposes. Velocity *subtracts*
   them because a wave travelling toward $+x$ with strain $\varepsilon$ carries
   particle velocity $-c_0\varepsilon$, while one travelling toward $-x$ carries
   $+c_0\varepsilon$. Then

   ```math
   v_\text{global} = s\,v_\text{local} + v_0,
   \qquad s = -1 \;\text{(input bar)}, \quad s = +1 \;\text{(output bar)}
   ```

   with $v_0$ added back because a uniformly translating bar carries no strain
   and so is invisible to every gauge.

3. **Specimen** (`specimen_response`): stress is the mean of the two faces, and
   strain comes from integrating the closing velocity:

   ```math
   \sigma = \frac{F_1 + F_2}{2A_s},
   \qquad
   \varepsilon(t) = \frac{1}{L_0}\int_0^{t} \bigl(v_\text{in} - v_\text{out}\bigr)\,dt'
   ```

   The printed $|F_1 - F_2| / \max|F_1|$ is the consistency check on that
   averaging.

### A consequence worth knowing

Because the bars are separate $2 \times 2$ problems that meet only in that final
averaging, **instrumentation cannot be traded between them**. A single gauge on
the input bar leaves $P_\text{in}$ and $M_\text{in}$ under-determined and
$F_\text{in}$ badly wrong, and no number of output-bar gauges appears anywhere in
the input bar's normal equations. Measured on the SHTB case, 1+2 and 1+3 agree to
three digits — both useless — while 2+2 is excellent. Spend gauges on the input
bar first.

## The two implementations, and why both are kept

There are two separation codes in this folder. This is deliberate, not leftover.

**`wave_separation.py` is the one to use.** It is the working library: `separate`
generalises the method to any number of gauges ≥ 2, and `backpropagate`,
`bar_interface`, `specimen_response`, `conditioning` and `single_wave_window`
carry the result through to specimen stress/strain. It validates its inputs, and
depends on nothing but numpy.

**`wsep.py` is a frozen reference.** It is a deliberately literal, 23-line
transcription of `wave_separation3` from `Prog_Treat/a05.m` — three gauges only,
MATLAB's argument order, no input checking, `n_fft` and `f_max` passed in by
hand. It exists to be read side by side with the MATLAB source.

Its purpose is to keep `sep_test.py`'s check on `wave_separation.py`
**independent**. Folding the two together would make that check circular: it
would prove only that the code agrees with itself. Because `wsep.py` was written
straight from the MATLAB and never refactored, their agreement is evidence.

Measured agreement is **7.1e-07 relative** (max |ΔP| = 6.6e-10 on a 9.3e-4
signal). The residual is one FFT bin: `wsep.py` reproduces MATLAB's
`ifft(..., 'symmetric')` on a zero-padded half spectrum, which *forces* the
Nyquist bin to zero, whereas `wave_separation.py` uses `rfft`/`irfft` and
computes it.

Two robustness differences, both in `wave_separation.py`'s favour and both
reasons not to use `wsep.py` for real work:

- with `eta = 0`, `wsep.py` divides by a zero determinant and returns all-NaN
  behind a RuntimeWarning; `separate` raises `ValueError`. Mismatched
  signal/position counts, duplicate or negative positions and non-uniform `t`
  are likewise unchecked in `wsep.py`;
- `separate` windows on `t - t[0]`, `wsep.py` on raw `t`. The exponential
  factors cancel for moderate offsets, but a record whose time base starts far
  from zero underflows: at `t + 2000 ms` with `eta = 1 /ms`, `wsep.py` returns
  NaN while `separate` is unaffected.

## Using the library on your own data

The reconstruction itself is `wave_separation.py`, which depends on nothing but
numpy — copy that one file wherever you need it. `reduce_specimen.py` is just a
driver for it; the core is these five lines:

```python
from wave_separation import separate, bar_interface, specimen_response

eps_p_in,  eps_m_in  = separate(t, [e1, e2, e3], [130.5, 530.5, 1176.5], c0=c0, eta=1.0)
eps_p_out, eps_m_out = separate(t, [e4, e5, e6], [129.5, 529.5, 1177.5], c0=c0, eta=1.0)

F_in,  v_in  = bar_interface(eps_p_in,  eps_m_in,  E, A, c0, outward=-1, v0=10.0)
F_out, v_out = bar_interface(eps_p_out, eps_m_out, E, A, c0, outward=+1, v0=0.0)

res = specimen_response(t, F_in, v_in, F_out, v_out, length=10.0, area=50.27)
```

The three parameters to get right for a different setup:

- **`positions`** — distance of each gauge from the bar/specimen interface, in
  the same length unit as `c0*t`. Order doesn't matter.
- **`outward`** — `-1` for the bar whose interior lies toward global −x, `+1` for
  the other. This only affects the sign of the returned velocity, not the force.
- **`v0`** — rigid-body velocity of the bar before impact. **`10.0` for this
  simulator's input bar; `0.0` for both bars of a classical SHPB.** This is the
  one that fails silently: omitting it leaves stress perfect and makes strain
  drift linearly.

For the real `a05` experiment you would additionally pass
`dispersion=(f, cp_over_c0)` from `Results_Raw/pochhammer.mat`, and use
`eta ≈ 500` because `t` is in seconds there rather than milliseconds.

## Choosing eta

`eta` is the exponential-window (Laplace) damping that regularizes the
separation. Stress needs no integration and is nearly insensitive to it; strain
is obtained by integrating velocity and is not:

| eta [1/ms] | stress err | strain err |
|---|---|---|
| 0.05 | 1.44e-2 | **3.86** |
| 0.2 | 1.40e-2 | 2.87e-1 |
| 0.5 | 1.41e-2 | 5.48e-3 |
| **1.0** | 1.47e-2 | **1.60e-3** |
| 5.0 | 1.75e-2 | 2.29e-3 |

Under-regularizing is far more dangerous than over-regularizing: at eta = 0.05
the stress looks perfect while the strain is garbage. **Tune eta on the strain,
never the stress.** The module docstring in `wave_separation.py` has the full
reasoning.

That table was measured on the direct-impact case. On the current SHTB
configuration the picture is blunter: eta is **completely flat from 0.5 to 8.0
/ms** — every error identical to four digits — and only degrades below 0.5. Once
you are above the floor, eta is not a tuning knob at all, and nothing is gained
by hunting for an optimum. Re-measure it for your own record rather than
assuming either table applies.

## Accuracy and time integration

Both simulators use explicit leapfrog on a lumped mass-spring chain. That is a
deliberate choice, not an unexamined default, and the error budget below is why.
The short version: **the time integration is not what limits accuracy here, and
neither an implicit scheme nor a spectral method would help.**

### The scheme is exactly non-dispersive at courant = 1

For this discretisation the leapfrog dispersion relation is

```
sin(w~ dt/2) = C · sin(k · dx/2),        C = c·dt/dx
```

At `C = 1` the sine inverts exactly, every wavenumber travels at exactly `c`, and
the numerical phase error is **zero at all wavelengths** — the "magic timestep".
Below it, the error is small and grows with frequency:

| f [kHz] | lambda [mm] | C = 0.5 | C = 0.8 | C = 1.0 |
|---|---|---|---|---|
| 10 | 505 | -4.8e-06 | -1.7e-06 | 0 |
| 100 | 51 | -4.8e-04 | -1.7e-04 | 0 |
| 200 | 25 | -1.9e-03 | -7.0e-04 | 0 |

(`c_num/c - 1`, for `dx = 1 mm`.) At `C = 0.5` and 100 kHz that is 0.29 µs of
timing drift over the full 3020 mm bar — 0.03 % of the 1097 µs pulse.

The consequence is the opposite of the usual intuition: **a larger timestep is
both faster and more accurate here.** Raising `courant` reduces phase error
rather than adding any.

### Where the error actually is

Measured on the SHTB case, varying one thing at a time:

| Variant | stress rel L2 | strain rel L2 |
|---|---|---|
| `C=0.8, damp=0.01, dx=1.0` (shipped) | 1.87e-03 | 2.27e-03 |
| `C=0.5` | 2.45e-03 | 2.29e-03 |
| half viscosity, `damp=0.005` | 5.50e-03 | 2.39e-03 |
| double viscosity, `damp=0.02` | 1.92e-03 | 2.46e-03 |
| 2x mesh, `dx=0.5` (2x cost) | 2.00e-03 | 2.11e-03 |
| **exact interface forces, no separation at all** | **1.82e-03** | — |

Two rows decide the question:

- **The last one is a ceiling test.** Hand the specimen estimator the
  simulator's own exact interface forces — skipping the wave separation
  entirely — and the stress error is still 1.82e-03, because the specimen is
  genuinely not in force equilibrium (face-force disagreement 2.75e-03). Most of
  the residual is there before any numerics.
- **Strain sits near 2.2e-03 regardless.** No integrator, viscosity or mesh
  setting moves it much, and eta does not move it at all. It is set by the 1D
  reduction — engineering strain from the face-velocity difference, against a
  specimen that is itself a wave-bearing body.

Note also that **artificial viscosity matters more than the integrator**:
halving it more than doubles the stress error. It is suppressing grid ringing at
the wavefronts, not merely cosmetic.

### Why courant = 0.8

| C | steps | stress | strain | F_in |
|---|---|---|---|---|
| 0.50 | 36861 | 2.45e-03 | 2.29e-03 | 5.25e-03 |
| **0.80** | **23038** | **1.87e-03** | **2.27e-03** | **4.39e-03** |
| 0.90 | 20478 | 1.68e-03 | 2.46e-03 | 4.18e-03 |
| 0.95 | 19400 | 1.63e-03 | 2.54e-03 | 4.14e-03 |
| 1.00 | — | NaN | | |

The magic timestep is unreachable in the tension model: the contact spring adds a
second force path at the anvil node and `C = 1.0` diverges, though it stays
stable to ~0.95. `0.8` takes 24 % off the stress error, leaves strain unchanged,
runs **1.6x fewer steps**, and keeps a 20 % stability margin. That margin is
robust — still stable at 0.9 with a steel striker, whose contact is 70x stiffer,
because `k_contact` derives from the striker's own modulus and `dt` already
tracks the fastest material present.

Curiously, the compression case is *worse* at the exact `C = 1.0, damp = 0`
setting (5.16e-02 against 4.23e-02 at `C = 0.5`). Exact integration then leaves
the discontinuous initial velocity as an undamped one-sample impulse at every
wavefront, and the reduction handles that ringing badly. Exact in time is not the
same as accurate overall.

### Why not implicit, why not spectral

**Implicit** (Newmark, generalised-α) buys unconditional stability, whose only
value is taking `dt` far above the CFL limit. For wave propagation that smears
exactly the wavefronts being measured — implicit schemes carry period elongation
and, with numerical damping, amplitude decay. It would also need a Newton solve
every step, because both the contact and the specimen plasticity are non-smooth.
Strictly worse, for real added complexity.

**Spectral** methods are a poor fit for this geometry. The model has material
discontinuities (steel anvil | aluminium bar | soft specimen), a unilateral
contact, a yield surface, and free ends. Fourier methods need periodicity and
smoothness and would ring at every one of those; Chebyshev handles the
non-periodicity but degrades the timestep limit to O(1/N²) and still breaks on
the discontinuities. A spectral *element* method with element boundaries placed
on the material interfaces would be the principled high-order choice — it is what
seismology uses, and it keeps an explicit leapfrog and a diagonal mass matrix —
but it is a full rewrite whose payoff is capped by the 1.8e-03 floor above.

**If more accuracy is ever needed, refine the mesh** — that is the lever with a
measurable effect (`dx = 0.5` gives 2.00e-03 at twice the cost) — or change the
specimen reduction, which is what the floor is actually made of.

## Files

| File | Purpose |
|---|---|
| `wave_separation.py` | **The library — use this one.** `separate`, `backpropagate`, `bar_interface`, `specimen_response`, `conditioning`, `single_wave_window`. numpy only. |
| `config.toml` | **All parameters, both cases** — materials, geometry, gauge locations, numerics, `eta`. |
| `config.py` | Reads and validates `config.toml`. stdlib `tomllib`, no dependency. |
| `recording.py` | Records only the gauge / interface / specimen rows. Resolves gauge distance → element, once, for both simulators. |
| `dump.py` | Writes and reads `dump.npz`. Its docstring lists every field. |
| `simulate.py` | 1D direct-impact COMPRESSION bar. Parameters from `[compression]`. |
| `simulate_tension.py` | 1D Split Hopkinson TENSION bar, POM striker tube and steel anvil. Parameters from `[tension]`. |
| `drive.py` | Runs `simulate.py`, writes `dump.npz`. Three lines — it sets nothing. |
| `drive_tension.py` | Same for `simulate_tension.py`. Writes the same filename — the two dumps overwrite each other. |
| `reduce_specimen.py` | Full chain: gauges → specimen stress/strain, validated against the simulator's own measurement. `--headless` to skip the window. |
| `plot_forces.py` | Raw gauge forces vs average specimen force. Shows when wave overlap begins. |
| `gauge_count_study.py` | How many gauges per bar are needed; compares 3+3, 2+3, 1+2, 1+1. |
| `sep_test.py` | Separation accuracy vs the simulator's interface force; eta sweep. |
| `wsep.py` | Frozen literal transcription of `wave_separation3`. Kept only as an independent cross-check — see above. Not for use. |
| `clean.sh` | Removes generated output and `__pycache__`. `-n` for a dry run. |

Generated at run time and safe to delete (`./clean.sh`): `dump.npz`,
`specimen.dat`, `specimen_reconstructed.dat`, `gauge_forces.png`,
`specimen_reconstruction.png`. `clean.sh` also removes the superseded
`eps.npy` / `force.npy` / `meta.npz` / `meta.npy` if an older run left them.

## Dependencies

`numpy` for the library and the simulators, `matplotlib` for the plots.
Nothing else: the config is read with `tomllib`, which is in the standard
library from Python 3.11.

## Validation

`reduce_specimen.py` reproduces the simulator's own specimen measurement from
the gauge signals alone, through 38 % strain including the wave-overlap regime:

```
peak specimen strain (true)      : 0.3851
peak strain rate (reconstructed) : 957 /s
stress   rel L2 err vs truth     : 3.7e-02
strain   rel L2 err vs truth     : 3.0e-03
force equilibrium |F1-F2|/max|F1|: mean 2.9e-02
```

That is the shipped 2-gauges-per-bar layout. The compression case is the one
that *loses* by dropping to two gauges — with three it gives 1.7e-02 stress and
1.6e-03 strain. The tension case gains; see below. If you work mainly in
compression, put `gauges = [130.0, 530.0, 1177.0]` back in `[compression]`.

## The tension simulator (SHTB)

`simulate_tension.py` models the Nicholas-type arrangement: a hollow striker
tube rides on the input bar, is launched **away** from the specimen, and strikes
an anvil at the bar's far end. That drags the bar after it, sending a tensile
pulse to the specimen. Because the striker surrounds the bar, the two occupy the
same range of x and are modelled as two independent chains coupled by a single
unilateral contact at the anvil — the striker can push it, never pull it.

The modelled rig, all of it set in `[tension]` of `config.toml`:

| Part | Material | Geometry |
|---|---|---|
| Input / output bar | 7075-T6 aluminium | ⌀16 mm, 3000 mm each |
| Striker | POM tube | ⌀16.1 / 40.0 mm, 800 mm, 10 m/s |
| Anvil | steel disc, rigid with the input bar | ⌀40 mm, 20 mm, 197 g |
| Specimen | elastic–plastic, JC hardening | ⌀5 mm, 10 mm |

Differences from the compression model that affect the reduction:

- **Tension positive.** The dump records this, and `specimen_response` is told
  by `reduce_specimen.py` rather than being hardcoded.
- **`v0 = 0` for both bars.** Only the striker moves, so there is no rigid-body
  velocity to add back — unlike the direct-impact case.
- **The specimen is bonded** to both bars (threaded, as a real tension specimen
  is), so the no-tension conditions at the specimen interfaces are gone. The
  unilateral condition lives at the striker/anvil contact instead.
- **The chain is not uniform.** Steel anvil, aluminium bars, soft specimen, so
  stiffness/area/density are per-element and nodal masses are assembled as half
  of each adjacent element's mass. Getting that wrong at the steel/aluminium
  junction would give the wrong reflection there.
- **The pulse length follows the STRIKER's wave speed**, not the bar's. POM is
  slow (1459 mm/ms against 5051 in the aluminium), so an 800 mm tube gives a
  **1097 µs** pulse — about 3.5× what the same tube in aluminium would.
- **Artificial viscosity is required.** The hard contact impact excites lattice
  ringing that never decays without it. `courant = 1.0` is unstable (NaN)
  because the contact spring adds a second force path at the anvil node.

Validated end to end, 2 gauges per bar, eta = 1.0 /ms, courant = 0.8:

```
peak specimen strain (true)      : 0.8009
peak strain rate (reconstructed) : 684 /s
stress   rel L2 err vs truth     : 1.9e-03
strain   rel L2 err vs truth     : 2.3e-03
force equilibrium |F1-F2|/max|F1|: mean 3.5e-03
```

The stress figure is within 3 % of the 1.82e-03 ceiling that the specimen
estimator gives even when handed *exact* interface forces — see
[Accuracy and time integration](#accuracy-and-time-integration). There is very
little left in this number that better wave separation could recover.

This is far better than the impedance-matched aluminium striker on 2000 mm bars
that preceded it (stress 6.7e-2, equilibrium 8.3e-2), and the long pulse is why:
1097 µs gives the specimen roughly nine transit times to equilibrate, so the two
face forces sit on top of each other and the separation is no longer competing
with a specimen that is genuinely out of equilibrium.

Note the record contains an unload/reload at ~1.9 ms, when the striker's release
wave returns — visible as a dip in the strain history and as extra branches in
the stress–strain curve. It is physical, not an artefact.
