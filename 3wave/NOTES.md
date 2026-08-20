# Working notes — bar calibration

Open threads and decisions from the calibration work of **2026-08-15**. Anything
already explained in `README.md`, `config.toml` or the script docstrings is *not*
repeated here; this file is only for what those do not record.

## Decisions taken

- **The coupler is 150 mm of bar stock at bar diameter.** Set in
  `[calibration_tension.specimen]`. The threaded connection's small impedance
  change is **deliberately neglected** — measured scaling says the bias goes as
  `L_joint * (1/c_bar - 1/c_joint)`, so a short thread engagement in matched
  material is second-order. A coupler of *different material* would not be, and
  fails silently; see "What a mismatched coupler costs" in the README.
- **`null_tol = 2.5e-3` is a coarse screen, not a precision check.** It separates
  a good calibration (1.21e-3) from a 0.9-ρ coupler (3.51e-3) by only 2.9×.
  A FAIL is conclusive; a PASS is weak evidence.
- ~~**The compression case is out of scope** for the calibration work.~~ **Done
  2026-08-18**, as a separate script rather than a case flag on the existing
  one. `identify_bar_compression.py` shares only the three signal-processing
  primitives; the method genuinely differs. Both far ends of a direct-impact rig
  are free, so each bar rings on its own round trip `2L/c`, that round trip is
  in the record, and the length closing the scale is the bar's own length rather
  than a gauge-to-free-end distance. Two bars, two speeds, two independent
  identifications — and the tape lands on the positions as a RELATIVE band,
  which the tension rig cannot manage. See "Calibrating a direct-impact bar" in
  `README.md`.
- **`specimen.length = 0` is now literal**, not a stand-in. The compression
  simulator meshes no element between the bars, `recording.py` takes the
  interface elements from the bar indices instead, `X_IN == X_OUT` on the
  contact plane, and the unilateral condition goes on ONE face rather than two —
  clipping both would stop the output bar carrying the tensile wave its own free
  end sends back, 1 mm inside the bar and for no physical reason.
- ~~**No loader for real experimental data yet**, by explicit decision.~~
  **Done 2026-08-20**, driven by `data/PC_bar_calibration.txt` — a real
  direct-impact shot into a polycarbonate bar. `experiment.py` builds the dump
  dict from a text record; `[experiment_pc_bar]` in `config.toml` holds the
  column map and the geometry; `config.py` validates that family through
  `_validate_experiment`, which drops the simulator-only tables.
  `identify_bar_compression.py --experiment CASE` runs on it unchanged in
  method. The keys a rig cannot supply — `c0_*`, `E_*`, `force_iface_*`,
  `spec_*` — are ABSENT rather than guessed, and that absence is what makes the
  true/error columns print a dash instead of quietly comparing against an input.
  See "Identifying a real, viscoelastic bar" in `README.md`.

- **`separate` grew an `attenuation` argument**, and polycarbonate is why. The
  complex wavenumber is `xi = (w - i eta)/c_p(f) - i alpha(f)`; `dispersion`
  keeps its old meaning (real `c_p`) and every simulated case is bit-identical
  at `attenuation=None`. `identify_attenuation.py` measures `alpha(f)` from the
  two-gauge transfer function — MAGNITUDES ONLY, never from a boundary
  condition, so the free-end null stays an independent check of it.

## Open threads

1. **The "It is good enough, end to end" table in `README.md` is stale and
   cannot be regenerated from anything checked in.** `reduce_specimen.py` reads
   `c0` and positions straight from the dump (simulator truth), so no script
   consumes the identified numbers. A scratch harness reproduced the ceiling row
   exactly (1.87e-03 / 2.27e-03) but gave 1.97e-03 / 3.28e-03 for the calibrated
   row against the table's 2.18e-03 / 3.70e-03 — the original run probably paired
   a 1 mm-joint calibration with a 10 mm-specimen shot and inherited a ~1 mm
   output-gauge offset from mesh rounding. **Do not patch the numbers; rebuild
   the comparison as a real script.** `--l-free-ref` now makes its ±2 mm sensitivity
   rows mechanically reproducible.
2. **The anvil figures are stale.** `README.md` and the `identify_bar_tension.py`
   docstring say a round-trip `c0` built on the anvil end comes out "4.1 % low".
   That is relative to the assembly length, which grew by 150 mm with the
   coupler, so it should now be nearer 4.0 %. Not re-measured. The +257 mm
   effective offset is a property of the anvil mass and should be unchanged.
3. **Tuning constants are still hardcoded in `identify_bar_tension.py`**, against
   this project's config-driven convention: `EDGE_MS = 0.12`, `TOL = 0.6*EDGE_MS`,
   `frac=0.3` in `_rise_index`, `n_want=4` in `candidates`, and the 1 % `Q`
   rejection band. `EDGE_MS` is the one that will bite on a real bar —
   it is tuned to this model's 59 µs rise, and Pochhammer–Chree dispersion
   widens edges. It could be derived from the measured 10–90 % rise instead.
4. **The `'  merged'` label in the per-gauge edge table is effectively dead
   code.** It requires all four candidates to fall within `TOL` of `P`. What
   actually rescues `out-0` is the `Q` outlier test, not the NaN path.

5. **TODO — port two fixes from `identify_bar_compression.py` back into
   `identify_bar_tension.py`.** Both were forced by the compression rig and both
   are strictly better than what the tension script does; neither has been
   applied there, so the tension results above are still the old behaviour.

   a. **Measured rise time instead of `EDGE_MS`.** This closes thread 3.
      `_rise_time(g, dt)` in `identify_bar_compression.py` takes the 10–90 %
      rise of the leading edge and the template is sized at `3 x rise`. The
      compression script had no choice — its polycarbonate edges are 5× broader
      than its aluminium ones, so one constant could not serve both bars. The
      tension script's `EDGE_MS = 0.12` is tuned to this model's 59 µs rise and
      is the constant most likely to bite on a real bar, where
      Pochhammer–Chree dispersion widens edges. `TOL = 0.6 * EDGE_MS` follows
      the template width and needs no separate thought.

   b. **Pick the STRONGEST candidate edge, not the EARLIEST.** The tension
      script's `candidates()` returns `(delay, value)` pairs and then takes
      `pick[0]`, the earliest surviving delay. A lumped-chain wavefront rings,
      so the cross-correlation carries sidelobes a few microseconds AHEAD of the
      true peak; on the compression record that put `f2` about 1.4 µs early and
      landed positions **4 mm out**, with every printed number still looking
      entirely plausible. `identify_bar_compression.py` uses
      `max(keep, key=lambda p: p[1])` instead. The tension script gets away with
      the earliest rule today only because its two features are far apart and
      its POM-driven edges are smooth — that is luck, not design.

   Porting is not a copy-paste: the tension script identifies ONE uniform
   assembly with a reference gauge and lags, so there is a single template and a
   single rise to measure, not one per bar. Re-run the "What comes out" table in
   `README.md` afterwards — the numbers there will move, and they are measured
   values, so regenerate them rather than editing them by hand.
6. **The polycarbonate bar identifies 6× worse than the aluminium one**
   (`c` −3.5e-03 against −5.5e-04), and it is the TIMESTEP, not the method:
   `dt` follows the fastest material present, so the polycarbonate elements run
   at an effective Courant number of 0.22 and disperse. A per-material timestep,
   or simply a finer mesh in the slow bar, would fix it. Not attempted.

7. **The +9 mm common position offset on the PC shot is not fully explained.**
   The identification recovers `D` to +0.88 mm but puts both gauges ~9 mm
   further from the impact face than the tape does. The attribution in
   `README.md` is the contact-end reflection at `2L/c`: the aluminium bar is
   still in contact there (its round trip is 944 us against the PC bar's 1460),
   so that reflection is not an ideal free surface and `f3` runs ~15 us late at
   every gauge alike. **Edge broadening over the extra `2x` of travel would also
   produce a positive offset**, and this record cannot separate the two — the
   broadening explanation predicts an offset ~4x larger at out-1 than at out-0
   and the measured ones are nearly equal (+8.55 / +9.42), which argues for the
   contact-end reflection, but not conclusively. **A shot with a SHORT striker
   settles it**: the bars then part long before `2L/c`, the contact end is a
   genuine free surface, and the offset should vanish. Worth one shot.

8. **`L = 1027 mm` was given as approximate and everything scales with it.**
   `(lengths, c) -> (lambda lengths, lambda c)` leaves the record invariant, so
   the identified `c`, `D` and positions all carry that error proportionally,
   and the free-end null is blind to it by construction. Nothing in the
   reconstruction is wrong because of it -- `separate` consumes transit times --
   but any absolute `c` or `E` quoted from this shot inherits it. Re-measure the
   bar.

9. **`alpha(f)` rests on one gauge pair.** Two gauges give one transfer
   function, so the fit has no redundancy and no way to tell material
   attenuation from anything else that broadens an edge with distance --
   Pochhammer-Chree dispersion in a 16.7 mm bar, or gauge-length averaging. A
   third gauge would give three pairs over three different baselines, and a
   disagreement between them would be the diagnostic. `fit_attenuation` already
   loops over all pairs; it just only has one here.

## Traps worth not rediscovering

- **The boundary conditions cannot pin `alpha`, only demand it.** Fitting
  `alpha` by minimizing the free-end null, or the tensile violation, or the
  post-separation residual, was tried and does not work: ALL of them improve
  monotonically as `alpha` is raised, with no minimum, because more damping
  quietly suppresses everything. They establish that `alpha > 0` is needed
  (0.124 tensile violation against 0.038) and they cannot choose its value. So
  `alpha` is measured from gauge magnitudes and the boundary conditions are left
  to validate it -- which is also the only way that validation means anything.

- **De-attenuation must be band-limited, and the failure looks like success.**
  The minus branch carries `exp(+alpha x)`. Let `alpha = k f` run to Nyquist and
  it overflows; stop just short and the free-end null residual reads ~15x BETTER
  than the truth, built entirely on amplified noise. The `(freq, alpha)` table
  form is its own band limit -- `np.interp` holds the endpoint value -- and
  `_pm_spectra` raises on the overflow case. The quiet case is the dangerous one.

- **A causality check needs the ECHO's rise, not a fixed clearance.** `M` must
  be zero at the contact until `2L/c`, but on the PC bar the echo has crossed
  2054 mm of lossy material and its 10-90 rise at the contact is 369 us. Scoring
  the check right up to `2L/c` reads 0.198 -- all of it the edge the check is
  waiting for. Measured from `M` itself it reads 0.050.

- **`c * lag` is not `D` in a lossy bar.** The far gauge sees a broader edge, so
  its correlation peak lands later and the arrival-lag route reads long: 374.86
  against 371.88 mm on the PC shot, where `f3 - f2` gives +0.88 mm against the
  tape. Both features of `f3 - f2` travelled the same path to the same gauge, so
  the bias cancels. It is ~1 us even on the simulated bars, from the lumped
  chain's numerical dispersion.

- **One shared edge is not always all of them.** The shared-delay detector used
  to find the single common event (the bars parting) and stop. A LONG striker --
  2415 mm of aluminium, 944 us round trip, against a 1027 mm bar -- is still in
  contact when the echo returns and puts shared edges at 944 and 1424 us. The
  second one survived the old detector and competed with the real echo on
  amplitude; it lost, but by luck. The detection is a loop now.

- **The free-end null must be windowed.** Over the full record it reads 1.2e-01,
  ~100× its true value, because the exponential window amplifies record-end
  truncation. A perfect calibration reports failure without `null_window`.
- **The `Q` spread cannot see a mismatched coupler.** The coupler's extra transit
  time enters `Q` identically at every gauge, so the spread is blind *by
  construction* — it even improves. The false bar asymmetry is the fingerprint.
- **`separate` positions are not scale-free.** `x = L_free +` a constant the tape
  error never touches, so a `L_free_ref` error moves positions *absolutely*
  (±1.3–2.0 mm here), not by the small relative figure. `c0`, `D` and `L_free` do
  scale. Several places in the docs used to claim otherwise.
- **Reconstructing at an arbitrary x must stay in the frequency domain.**
  `separate` ends with `irfft(X, n_fft)[:n]`, truncating 131072 samples to
  23038; the discarded tail is not zero, so re-`rfft`ing its time-domain output
  and phase-shifting that is a *different signal*. Measured, reproducing the
  recorded gauge strain: **1.4e-01** that way against **9.3e-15** keeping the
  P/M spectra. `separate_field` exists so the wrong route is not reachable — it
  never lets the spectra escape. The wrong version looks entirely plausible
  plotted, which is the whole danger.
- **The free-end null is a boundary layer, not a property of "the end region".**
  It holds *at* the surface: 3.3e-04 there, 3.3e-03 ten mm in, 3.0e-02 at
  100 mm, because just inside the surface the two waves are offset by `2x/c0`
  and stop cancelling. A station grid coarser than a few mm will not show it, so
  `lagrange_diagram.py` evaluates the exact end station separately for the
  printed metric rather than reading it off the image grid. Unwindowed it reads
  6.5e-02 *everywhere* — same `exp(+eta t)` trap as the identification.
- **`L_free_*` is not the validity limit of a reconstruction; `L_bar_*` is.**
  `L_free_in` is 3020 mm on the SHTB but 20 mm of that is the steel anvil, and
  `separate` assumes one wave speed the whole way. `recording.py` now derives
  both and the dump carries them. It **clamps** `L_bar_*` to `L_free_*` on
  purpose: `simulate_compression.py` builds its bar indices as NODE indices and
  uses them on element arrays (its own comment says so), so `out_e.max() + 1`
  runs one element past the model and would otherwise report a bar longer than
  the distance to its own end.
- **`L_free` is a length; `xi` is a wavenumber. They are not the same symbol.**
  Until **2026-08-16** the calibration's gauge-to-free-end distances were called
  `xi` / `xi_ref`, colliding with the separation's complex wavenumber
  `xi = (w - i eta)/c_p` — which appeared, in the *other* sense, in the same
  `identify_bar_tension.py` docstring. Renamed to `L_free*` throughout the
  calibration path (config keys, CLI flags, prose); `wave_separation.py` and
  `wsep.py` keep `xi` for the wavenumber and were not touched. The name matches
  the dump's existing `L_free_in` / `L_free_out`, which are the same distance
  measured to a bar face instead of to a gauge.
