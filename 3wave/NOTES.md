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
- **No loader for real experimental data yet**, by explicit decision. `--xi-ref`
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
   the comparison as a real script.** `--xi-ref` now makes its ±2 mm sensitivity
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
- **`separate` positions are not scale-free.** `x = xi +` a constant the tape
  error never touches, so a `xi_ref` error moves positions *absolutely*
  (±1.3–2.0 mm here), not by the small relative figure. `c0`, `D` and `xi` do
  scale. Several places in the docs used to claim otherwise.
