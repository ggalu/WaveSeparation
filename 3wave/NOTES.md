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
- **The compression case is out of scope** for the calibration work.
  `identify_bar_tension.py` hardcodes `config.load('calibration_tension')`. If a
  `calibration_compression` counterpart is ever wanted, the case name has to
  become a CLI argument too.
- **No loader for real experimental data yet**, by explicit decision. `--l-free-ref`
  is the conceptually load-bearing half of rig readiness; the other half is a
  loader that builds the dump dict from scope traces, since `eps_in`/`dt`/`N`
  are real on a rig but `pos_in` is not.

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

## Traps worth not rediscovering

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
