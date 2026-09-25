# Hopkinson Bar Wave Separation — Discussion

This document collects the source theory writeup and the follow-up discussion on
why the windowed FFT buys numerical stability in the wave-separation method.

---

## Source: Theory writeup

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

$$
\varepsilon_k(\omega) = \underbrace{P(\omega)\,e^{-i\xi x_k}}_{\text{away from specimen}}
             + \underbrace{M(\omega)\,e^{+i\xi x_k}}_{\text{toward specimen}}
$$

where

$$
\xi = \frac{\omega - i\eta}{c_p}
$$

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

$$
\mathcal{L}\{f\}(s) = \int_0^\infty f(t)\,e^{-st}\,dt
                    = \int_0^\infty \underbrace{f(t)\,e^{-\eta t}}_{\text{the window}}
                      e^{-i\omega t}\,dt
                    = \mathcal{F}\bigl\{f\,e^{-\eta t}\bigr\}(\omega)
$$

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

$$
e^{-s x / c_p} = e^{-(\eta + i\omega)x/c_p} = e^{-i\xi x},
\qquad
\xi = \frac{\omega - i\eta}{c_p} = \frac{s}{i\,c_p}
$$

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

$$
\begin{bmatrix} a_1 & b_1 \\ a_2 & b_2 \end{bmatrix}
\begin{bmatrix} P \\ M \end{bmatrix}
=
\begin{bmatrix} \varepsilon_1 \\ \varepsilon_2 \end{bmatrix},
\qquad
\mathbf{A}\mathbf{z} = \boldsymbol{\varepsilon}
$$

The determinant collapses to a single sine. With $D = x_2 - x_1$ the gauge
spacing,

$$
\det\mathbf{A} = a_1b_2 - b_1a_2
       = e^{+i\xi D} - e^{-i\xi D}
       = 2i\,\sin(\xi D)
$$

so Cramer's rule gives the separated waves outright:

$$
P = \frac{\varepsilon_1 e^{+i\xi x_2} - \varepsilon_2 e^{+i\xi x_1}}{2i\,\sin(\xi D)},
\qquad
M = \frac{\varepsilon_2 e^{-i\xi x_1} - \varepsilon_1 e^{-i\xi x_2}}{2i\,\sin(\xi D)}
$$

**Everything about the method is already visible here.** The solution exists
unless $\sin(\xi D) = 0$, and only the spacing $D$ appears — not where the pair
sits on the bar. Splitting $\xi D$ into its real and imaginary parts,

$$
\bigl|\sin(\xi D)\bigr|^2
  = \sin^2\!\left(\frac{\omega D}{c}\right) + \sinh^2\!\left(\frac{\eta D}{c}\right)
$$

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

$$
\min_{P,\,M}\; J(P,M), \qquad
J = \bigl\|\boldsymbol{\varepsilon} - \mathbf{A}\mathbf{z}\bigr\|^2
  = \sum_{k=1}^{K} \bigl| \varepsilon_k - P\,a_k - M\,b_k \bigr|^2
$$

$J$ is real and quadratic, so its minimum is where the derivatives with respect
to $\bar{P}$ and $\bar{M}$ both vanish:

$$
\frac{\partial J}{\partial \bar{P}}
 = -\sum_k \bar{a}_k\bigl(\varepsilon_k - P a_k - M b_k\bigr) = 0,
\qquad
\frac{\partial J}{\partial \bar{M}}
 = -\sum_k \bar{b}_k\bigl(\varepsilon_k - P a_k - M b_k\bigr) = 0
$$

Rearranging each into unknowns-on-the-left form, and writing
$\langle \mathbf{u},\mathbf{v}\rangle = \sum_k \bar{u}_k v_k$,

$$
P\,\langle \mathbf{a},\mathbf{a}\rangle + M\,\langle \mathbf{a},\mathbf{b}\rangle
  = \langle \mathbf{a},\boldsymbol{\varepsilon}\rangle,
\qquad
P\,\langle \mathbf{b},\mathbf{a}\rangle + M\,\langle \mathbf{b},\mathbf{b}\rangle
  = \langle \mathbf{b},\boldsymbol{\varepsilon}\rangle
$$

which is $\mathbf{A}^{H}\mathbf{A}\,\mathbf{z} = \mathbf{A}^{H}\boldsymbol{\varepsilon}$ —
**the $K$ equations have collapsed back into a $2 \times 2$ system:**

$$
\begin{bmatrix}
\langle \mathbf{a},\mathbf{a}\rangle & \langle \mathbf{a},\mathbf{b}\rangle \\
\langle \mathbf{b},\mathbf{a}\rangle & \langle \mathbf{b},\mathbf{b}\rangle
\end{bmatrix}
\begin{bmatrix} P \\ M \end{bmatrix}
=
\begin{bmatrix}
\langle \mathbf{a},\boldsymbol{\varepsilon}\rangle \\
\langle \mathbf{b},\boldsymbol{\varepsilon}\rangle
\end{bmatrix}
$$

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

$$
h_1 = \sum_k e^{-2\eta x_k/c} > 0,
\qquad
h_2 = \sum_k e^{+2\eta x_k/c} > 0,
\qquad
g   = \sum_k e^{+2i\omega x_k/c}
$$

so $h_1$ and $h_2$ are real and positive while $g$ is a sum of pure phases, and
Cramer's rule finishes it with

$$
\det = h_1 h_2 - |g|^2
$$

#### The two routes are the same route

For $K = 2$ the least-squares machinery does **not** give a different answer from
the exact solve above. $\mathbf{A}$ is then square and invertible, so

$$
\mathbf{z} = \bigl(\mathbf{A}^{H}\mathbf{A}\bigr)^{-1}\mathbf{A}^{H}\boldsymbol{\varepsilon}
           = \mathbf{A}^{-1}\bigl(\mathbf{A}^{H}\bigr)^{-1}\mathbf{A}^{H}\boldsymbol{\varepsilon}
           = \mathbf{A}^{-1}\boldsymbol{\varepsilon}
$$

The residual $J$ is zero and the fit is an interpolation. Numerically, the
explicit Cramer expressions above and `separate` agree to $6 \times 10^{-14}$
relative on the shipped 2-gauge dump.

The determinants match too, which is worth noting because the next subsection
leans on it: for a square $\mathbf{A}$,
$\det(\mathbf{A}^{H}\mathbf{A}) = |\det\mathbf{A}|^2$, so

$$
h_1 h_2 - |g|^2 = \bigl|2i\sin(\xi D)\bigr|^2 = 4\bigl|\sin(\xi D)\bigr|^2
$$

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

$$
\det = 2\left[\, \cosh\!\left(\frac{2\eta D}{c}\right)
                   - \cos\!\left(\frac{2\omega D}{c}\right) \right]
$$

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
a layout before committing to it.

### From the separated waves to the force at the specimen

The two bars are solved **completely independently**; they share nothing until
the very last step. For a 2+2 layout the chain runs:

1. **Per bar** (`separate`): two gauge signals → windowed FFT → for each
   frequency bin, solve that $2 \times 2$ system → $P(\omega), M(\omega)$ →
   inverse FFT → undo the window with $e^{+\eta t}$ → $\varepsilon_+(t)$ and
   $\varepsilon_-(t)$, both now at $x = 0$.

2. **Interface state** (`bar_interface`):

   $$
   F = EA\,(\varepsilon_+ + \varepsilon_-),
   \qquad
   v_\text{local} = c_0\,(\varepsilon_- - \varepsilon_+)
   $$

   Force *adds* the two waves because strain superposes. Velocity *subtracts*
   them because a wave travelling toward $+x$ with strain $\varepsilon$ carries
   particle velocity $-c_0\varepsilon$, while one travelling toward $-x$ carries
   $+c_0\varepsilon$. Then

   $$
   v_\text{global} = s\,v_\text{local} + v_0,
   \qquad s = -1 \;\text{(input bar)}, \quad s = +1 \;\text{(output bar)}
   $$

   with $v_0$ added back because a uniformly translating bar carries no strain
   and so is invisible to every gauge.

3. **Specimen** (`specimen_response`): stress is the mean of the two faces, and
   strain comes from integrating the closing velocity:

   $$
   \sigma = \frac{F_1 + F_2}{2A_s},
   \qquad
   \varepsilon(t) = \frac{1}{L_0}\int_0^{t} \bigl(v_\text{in} - v_\text{out}\bigr)\,dt'
   $$

   The printed $|F_1 - F_2| / \max|F_1|$ is the consistency check on that
   averaging.

---

## Q1 — Why multiplying by $e^{-\eta t}$ before an FFT genuinely equals a Laplace transform, and what "stepping off the real frequency axis" buys you numerically

The identity is exact, and it's worth seeing that there's genuinely no
approximation in the *equality* itself — the approximation lives only in the
discretization. Start from the one-sided Laplace transform with $s$ complex, and
split $s$ into real and imaginary parts, $s = \eta + i\omega$:

$$
\mathcal{L}\{f\}(s) = \int_0^\infty f(t)\,e^{-st}\,dt = \int_0^\infty f(t)\,e^{-(\eta + i\omega)t}\,dt.
$$

Now do nothing but regroup the exponential:

$$
e^{-(\eta+i\omega)t} = e^{-\eta t}\,e^{-i\omega t} \quad\Longrightarrow\quad \mathcal{L}\{f\}(\eta + i\omega) = \int_0^\infty \big[\,f(t)\,e^{-\eta t}\,\big]\,e^{-i\omega t}\,dt.
$$

The bracketed quantity is just a new time signal, $g(t) = f(t)\,e^{-\eta t}$, and
what's left, $\int_0^\infty g(t)\,e^{-i\omega t}\,dt$, is the Fourier transform of
$g$ (one-sided is fine because the record is causal — zero before $t=0$). So

$$
\mathcal{L}\{f\}(\eta + i\omega) = \mathcal{F}\big\{f \cdot e^{-\eta t}\big\}(\omega).
$$

That's the whole content. Whether you attach the real exponential to $f$ (calling
it a "window") or to $e^{-i\omega t}$ (calling it "the real part of $s$") is a
bookkeeping choice; the integrand is the same. `np.fft.rfft` computes the
discretized $\mathcal{F}$ with exactly the $e^{-i\omega t}$ sign convention, so
premultiplying your samples by $e^{-\eta t}$ turns that same routine into the
discretized $\mathcal{L}$. The only gap between the two sides is that the FFT is a
finite rectangle-rule sum over $[0,T]$ rather than the integral over
$[0,\infty)$ — which is precisely why the doc's verification against $f = e^{-at}$
recovers $1/(s+a)$ to only $2\times10^{-4}$ and not to machine precision. That
residual is discretization error, not a conceptual defect.

**The geometric picture.** In the complex $s$-plane, a pure Fourier transform
samples $\mathcal{L}$ along the imaginary axis: as $\omega$ sweeps, you march up
and down $\mathrm{Re}(s) = 0$. Choosing $\eta > 0$ slides that entire sampling
line rightward to $\mathrm{Re}(s) = \eta$ — the Bromwich contour. The FFT still
supplies the vertical $i\omega$ sweep; the window $e^{-\eta t}$ supplies the
horizontal offset. "$\xi$ is that same $s$ in disguise" because a wave
propagating at $c_p$ carries the factor $e^{-sx/c_p} = e^{-i\xi x}$ with
$\xi = s/(i c_p)$, so the offset $\eta$ you put into the window is the same $\eta$
that appears in the imaginary part of the wavenumber.

**What stepping off the axis buys you.** This is where it stops being bookkeeping
and becomes the point of the method. The separation requires inverting, at each
frequency bin, a system whose two-gauge determinant is $2i\sin(\xi D)$. With
$\xi = (\omega - i\eta)/c_p$, split the modulus into real and imaginary
contributions:

$$
|\sin(\xi D)|^2 = \sin^2\!\left(\frac{\omega D}{c}\right) + \sinh^2\!\left(\frac{\eta D}{c}\right).
$$

On the pure Fourier axis, $\eta = 0$, the second term vanishes and you're left
with a real sine that passes *through zero* every time $\omega D/c = n\pi$ —
whenever the gauge spacing is a whole number of half-wavelengths. At those
frequencies the matrix is exactly singular: the two gauges see identical phase,
$P$ and $M$ are not separable, and the reconstruction divides by (something
arbitrarily near) zero. These aren't rare pathological points; for the shipped
$D = 400$ mm they recur every 6.31 kHz right across the band you care about.

The $\sinh^2(\eta D/c)$ term is what fixes this, and notice its two properties: it
is strictly positive for $\eta > 0$, and it is **independent of $\omega$**. So it
sets a floor under the determinant that the sine's zeros can no longer reach.
Every frequency becomes invertible; the singularities are lifted off zero.

Why moving into the complex plane accomplishes that is worth stating physically,
because it's not an arbitrary numerical patch. On the imaginary axis the two
counter-propagating waves differ only in *phase*, and phase shifts can coincide —
that's exactly the half-wavelength degeneracy. Off the axis, $\eta > 0$ gives each
wave a real spatial factor $e^{\pm \eta x/c}$: one wave grows along the bar, the
other decays. That is an *amplitude* asymmetry between "toward specimen" and "away
from specimen," and it is frequency-independent, so it can never be masked by a
phase coincidence. The $\sinh$ term is the algebraic fingerprint of that amplitude
difference. This is the precise sense in which "the regularisation and the Laplace
transform are the same act": choosing the Bromwich offset and lifting the
determinant off zero are one choice with one mechanism, not two.

**The catch, so the tradeoff is clear.** You cannot simply crank $\eta$ up to make
conditioning arbitrarily good, because the inversion has to undo the window by
multiplying by $e^{+\eta t}$. That factor grows, so anything living at the tail of
the record — noise, truncation residue — gets amplified by $e^{+\eta T}$ over the
record length $T$. Hence the doc's table and the $\eta T$ ceiling: too small an
$\eta$ leaves the determinant near-singular at the bad frequencies, too large an
$\eta$ blows up the tail on inversion, and the usable window is where $\eta$ is big
enough to lift the determinant meaningfully while $\eta T$ stays modest (their
table shows it's already unusable by $\eta T \approx 30$). The forward window
incidentally also tapers the end of the record, which makes the finite-$T$ FFT a
cleaner approximation to the semi-infinite integral — but the inversion
re-amplifies exactly that tail, so you don't get that suppression for free in the
round trip.

---

## Q2 — Re-explain without resorting to the Laplace transform: why a windowed Fourier transform buys numerical stability

Drop Laplace entirely and stay inside the Fourier picture the whole way. The trick
is to ask what the window does to the *model*, not to the transform.

**What the window does to a traveling wave.** Each gauge signal is a sum of two
propagating pieces: a wave heading into the bar, which reaches gauge $k$ as a
*delayed* copy of the interface signal, and a wave heading back out, which reaches
gauge $k$ as an *advanced* copy. Write the outgoing piece as $p(t - x_k/c)$. Now
form the windowed Fourier transform and substitute $\tau = t - x_k/c$:

$$
\int p(t - x_k/c)\,e^{-\eta t}\,e^{-i\omega t}\,dt = e^{-\eta x_k/c}\,e^{-i\omega x_k/c}\int p(\tau)\,e^{-\eta\tau}\,e^{-i\omega\tau}\,d\tau.
$$

Look at what fell out front. The delay contributes the usual phase
$e^{-i\omega x_k/c}$, but *also* a real factor $e^{-\eta x_k/c}$. The reason is
entirely mechanical: by the time the delayed copy arrives, the window has already
decayed, so the delayed copy is attenuated. The advanced (returning) wave
$m(t + x_k/c)$ does the opposite and picks up a real *growth* factor
$e^{+\eta x_k/c}$. So the windowed model is

$$
\tilde\varepsilon_k(\omega) = P\,\underbrace{e^{-\eta x_k/c}}_{\text{real}}\,e^{-i\omega x_k/c} + M\,\underbrace{e^{+\eta x_k/c}}_{\text{real}}\,e^{+i\omega x_k/c},
$$

and if you insist on packing each pair back into one exponential you get exactly
$e^{\mp i\xi x_k}$ with $\xi = (\omega - i\eta)/c$. The complex wavenumber is not a
Laplace artifact; it is just "a delayed copy of a decaying signal is attenuated,"
which is pure Fourier bookkeeping.

The compact way to say the same thing, if you like: multiplying by $e^{-\eta t}$
in time evaluates the spectrum at the shifted argument $\omega - i\eta$. That is
the modulation/shift property of the Fourier transform — an analytic continuation
of the same transform off the real $\omega$ axis, no new machinery.

**Why that stabilizes the solve.** The separation at each frequency asks you to
distinguish the two waves given their values across the gauge array. The two
"signatures" are the column vectors

$$
\mathbf{a} = \big(e^{-\eta x_k/c}\,e^{-i\omega x_k/c}\big)_k, \qquad \mathbf{b} = \big(e^{+\eta x_k/c}\,e^{+i\omega x_k/c}\big)_k,
$$

and the system is singular exactly when these are parallel, because then no
combination of the two can be told apart. The clean test is the per-gauge ratio

$$
\frac{a_k}{b_k} = e^{-2\eta x_k/c}\;e^{-2i\omega x_k/c}.
$$

Parallel means this ratio is the *same at every gauge*. Now split it into
magnitude and phase. With $\eta = 0$ the magnitude is $1$ at every gauge, so the
ratio is pure phase and depends on $x_k$ only through $e^{-2i\omega x_k/c}$. That
phase can come back into step across the array — precisely at the
half-wavelength-commensurate frequencies — and when it does, the ratios coincide,
the columns align, and the determinant $\propto \sin(\omega D/c)$ passes through
zero. The two waves are genuinely indistinguishable there because at those
frequencies they present the same phase at every gauge and nothing else separates
them.

Turn on $\eta$ and the magnitude of the ratio becomes $e^{-2\eta x_k/c}$, which
*varies monotonically with gauge position*. Two distinct gauges now have ratios of
different magnitude, so the ratios can never coincide, whatever the phases are
doing. The columns cannot be parallel, and the determinant is held off zero by a
floor $\sinh^2(\eta D/c)$ that carries no $\omega$ in it. That $\omega$-independence
is the whole point: a phase coincidence is a frequency-specific event, and you are
defeating it with an amplitude difference that exists at every frequency, so no
phase alignment can ever mask it.

The physical reading is worth keeping in mind because it is what makes this not
feel like a numerical trick. On the real axis the two counter-propagating waves
differ only in the *direction their phase winds* along the bar, and directions of
winding can momentarily agree. The window imposes, on top of that, a real spatial
tilt: the outgoing signature decays down the bar and the incoming one grows. That
is an amplitude asymmetry between "toward the specimen" and "away from it," and
amplitude is not something a phase coincidence can counterfeit.

**One thing to keep separate.** This is not the leakage-reduction role you
normally associate with windowing. A Hann or similar taper is symmetric and exists
to soften both ends of the record against spectral leakage. This exponential
window is one-sided and its job is the analytic continuation above: it tilts the
two wave signatures apart in amplitude so the per-frequency solve stays invertible.
It does incidentally taper the tail, which makes the finite-length FFT a slightly
cleaner stand-in for the infinite integral, but that is a side effect, not the
reason it is there.

And the tradeoff you already saw survives this retelling unchanged: recovering the
physical waves means dividing the window back out, i.e. multiplying by
$e^{+\eta t}$, which regrows the record's tail by $e^{+\eta T}$. So $\eta$ has to
be large enough to tilt the signatures apart at the bad frequencies yet small
enough that $\eta T$ stays modest, which is the ceiling their table quantifies.
