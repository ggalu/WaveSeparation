# User's guide: from a calibration shot to the force on a specimen

This guide takes a measured experiment from raw gauge records to the force at the
specimen. It covers the two rigs this project supports:

- **Part A — Split Hopkinson tension bar (SHTB).** Both bars are instrumented. You
  get the force on both sides of the specimen.
- **Part B — Direct-impact pressure bar.** The striker is the input bar and only
  the output bar carries gauges. You get the force on the output side.

Both parts follow the same two steps:

1. **Identify the bars once**, from a *calibration shot* with no specimen. This
   measures the wave speed `c0`, the gauge positions, and the bar's attenuation
   α(f) and dispersion c_p(f). They are properties of the bars, not of a shot.
2. **Analyse every specimen shot** with those identified properties. Nothing is
   identified from a specimen shot, because the specimen smooths away the sharp
   edges the identification times.

The theory and the evidence for each choice are in `README.md`. This guide only
says what to do and what to look at.

---

## 0. Before you start

**Software.** Python ≥ 3.11 (`tomllib`), `numpy` and `matplotlib`. Run everything
from the `3wave/` folder.

**Where things are.** The scripts in this guide are the `*.py` files at the top
of `3wave/`, and you run them from there. `wave_separation_code/` is the library
they import, and `simulation_code/` is only needed for synthetic shots.

**Case folders.** Every step is a folder under `cases/`, holding its own
`case.toml`, its input record and everything the scripts write for it:

```
cases/identifications/<name>/   calibration shot  -> bar_identified.npz
cases/analyses/<name>/          specimen shot     -> interface_force.{png,dat}
```

Every script takes the folder as its one argument, and reads and writes only
there. An analysis names the identification it uses with its `bars` key.
Settings every case shares (`eta`) live in `defaults.toml`, and any case may
override them.

**The record file.** Plain whitespace-separated text that `numpy.loadtxt` can read:

- one **time column in microseconds**, uniformly sampled;
- one column per gauge, in **force** (kN by default; set `units = "N"` for
  newtons). Strain or volts work too: the separation is linear, so anything
  proportional to strain goes in, and the same quantity comes out. No E, A or ρ
  is needed to reconstruct the force.

The loader subtracts each channel's pre-trigger mean. It then trims the record
to just ahead of the first arrival and re-zeros t. The output `.dat` files give
time in the **source file's** own base, so you can compare them with the scope
trace directly.

**Three rules for any record:**

1. **Quiet at the start.** The record must be at rest where the analysis starts.
   Keep enough pre-trigger. Never cut the record just before the interesting
   part: that leaves a standing load in the bar, and the separation cannot model
   it.
2. **No clipping.** A channel that hits the amplifier rail is reported, but it
   is not modelled. Set the gain so that the peaks stay well clear of the rail.
   `identifications/tension_bar` is a shot lost to this problem, and
   `tension_bar_2` is the same shot re-recorded.
3. **Long enough.** A calibration shot must contain the free-end echoes the
   identification times (the length is given per rig below).

---

# Part A — Split Hopkinson tension bar (SHTB)

The rig: a striker loads the input bar through an anvil. The specimen sits
between the input bar and the output bar, and the output bar's far end is free.
Each bar carries two gauges, named `in-0`, `in-1`, `out-0` and `out-1`. `x` is
measured from each bar's face at the specimen, positive into the bar.

## A1. Fire the calibration shot

- Bolt the two bars together **through a coupler of bar material**, with no
  specimen. The assembly then acts as one long bar with a single clean reflector,
  the output bar's free end.
- Record long enough to see **both out-bar gauges' free-end echoes**: at least
  one assembly round trip, 2·(L_input + joint + L_output)/c0, after the first
  arrival. On the current rig that is about 2.2 ms, so a record of 5 ms is
  comfortable.
- Keep about 1 ms of pre-trigger.

## A2. Measure with a tape

| what | key | how well it matters |
|---|---|---|
| every gauge's distance from its bar face | `gauges` | ±2 mm is normal. The out-bar **spacing** is what the recommended route anchors on. |
| input bar length | `[input_bar] L_input` | used for checks |
| output bar length | `[output_bar] L_output` | used for checks |
| coupler length | `[specimen] length` | used for checks |
| one long length: a gauge to the output bar's far free end | `L_free_ref`, `L_free_ref_gauge` | the scale for the default route; a check under `out_echo_diff` |
| bar diameter | `diameter` | reporting only |

Only one length sets the scale of everything, because a record fixes only
lengths/c0. Which length it is depends on `c0_route` (see A4). Everything else
you measure becomes an independent check.

## A3. Create the identification folder

```bash
mkdir cases/identifications/my_shtb_cal
cp /path/to/record.txt cases/identifications/my_shtb_cal/
```

Write `cases/identifications/my_shtb_cal/case.toml`. This template follows
`cases/identifications/tension_bar_2/case.toml`, which is fully commented:

```toml
kind = "identification"
method = "tension"               # -> identify_bar_tension.py

loading = "tension"              # sign convention of the record: tension positive
data = "record.txt"
units = "N"                      # omit for kN
gauges = [935.0, 119.0, 123.0, 1200.0]   # TAPE, same order as [columns]

L_free_ref = 3730.0              # tape: L_free_ref_gauge -> output bar's far free end
L_free_ref_gauge = "in-0"
L_free_ref_tol = 5.0             # what that tape is good to [mm]
c0_route = "out_echo_diff"       # see A4
gauge_tol = 2.0                  # what ONE tape gauge position is good to [mm]

[columns]                        # 0-based column indices in the record
time = 0
"in-0"  = 1                      # gauge names MUST start with in- / out-
"in-1"  = 2
"out-0" = 3
"out-1" = 4

[input_bar]
L_input = 2760.0
diameter = 16.0

[output_bar]
L_output = 2777.0
diameter = 16.0

[specimen]
length = 23.0                    # the coupler, for this shot

[trim]
threshold = 0.05                 # fraction of peak that counts as "arrived"
lead = 500.0                     # us of quiet record kept before the arrival
baseline = true

[attenuation]                    # fit alpha(f) and c_p(f) from the out-bar pair
f_lo = 2.0                       # kHz
f_hi = 50.0                      # kHz; also the band limit
snr = 0.01

[null]
window = 0.75
tol = 2.5e-3                     # pass threshold for the free-end null
```

## A4. Run the identification

```bash
python3 identify_bar_tension.py cases/identifications/my_shtb_cal
```

Add `--headless` to write the figure without opening a window. To test how
sensitive the result is to the tape, override the reference length from the
command line: `--l-free-ref 3735 --l-free-ref-tol 5`.

The script writes `bar_identified.npz`, which the analysis reads, and
`bar_identification.png`.

**Choosing `c0_route`.** This is the one real decision.

- `"out_echo_diff"` is recommended when both out-bar echoes are clean. c0 comes
  from the difference of the two out-bar free-end round trips, so its only
  anchor is the out-bar gauge spacing. It never crosses the coupler, and the
  identified out-bar positions become a real check against the tape. Before you
  trust it, check that rows (3) and (4) of the c0 table agree. On
  `tension_bar_2` they agree to 1e-3.
- `"joint"` is the default. It uses `L_free_ref` and averages over every gauge,
  so a single bad echo is rejected. Any extra transit time through the coupler
  is absorbed into c0.
- `"out_echo"` exists only to reproduce old results. Do not use it.

**Reading the printout, from top to bottom:**

1. **Edges per gauge.** Each gauge needs an arrival and a `2L_free/c0` echo.
   A missing echo means the record is too short or clipped.
2. **The c0 table.** Rows (2) and (5) measure the same spacing along different
   paths. A 1 % gap between them is the edge changing shape with distance, not
   an error in geometry.
3. **"What the out-bar anchor implies".** This section compares your tape bar
   lengths with the anchor. A large `joint, acoustic` slip, 68 mm on the current
   rig, means the coupler is not acoustically bar stock, or a bar length is
   wrong.
4. **Gauge positions.** The identified `x` is shown next to the tape value,
   with a tolerance per gauge.
   - A common offset shared by a bar's gauges is harmless: it moves the plane
     that is reconstructed, but not the shape of the force.
   - A spread between the gauges is not harmless.
   - If one gauge is far off its tape value and you trust the tape more, pin it
     under `[position_override]`. `tension_bar_2` does this for `in-1`, and its
     comments explain why.
5. **Attenuation and dispersion.** α(f) and c_p(f)/c0 are fitted from the
   output-bar pair and applied to both bars (one cylinder). Look at the line
   "far gauge predicted from near": it should drop from *lossless* to
   *alpha+dispersion*.
6. **Free-end null.** This check uses no ground truth. It reconstructs the
   stress at the output bar's free surface, which must be zero.
   - A **FAIL is conclusive**: the transit times are wrong, usually because of
     the coupler or a bar length.
   - A **PASS is weak evidence**: the test cannot see a common scale error.
   - `tension_bar_2` currently prints FAIL (2.5e-2 against a threshold of
     2.5e-3). That is consistent with the joint slip in item 3, and the rig's
     open threads are in `NOTES.md`.
7. **Ready to use.** The c0 and the gauge positions that were written to
   `bar_identified.npz`.

## A5. Optional: check the calibration shot itself

This is worth running once on every new calibration:

```bash
python3 reconstruct_interface.py cases/identifications/my_shtb_cal   # F at each bar face, equilibrium, null
```

A rigid coupler carries the same force on both sides, so `F_in ≈ F_out` is the
check. `identify_bar_tension.py` already prints it as its last line ("force
equilibrium across the coupler"). On `tension_bar_2` the mean mismatch is 2.8 %
of peak.

## A6. Create the analysis folder for a specimen shot

```bash
mkdir cases/analyses/my_specimen
cp /path/to/specimen_record.txt cases/analyses/my_specimen/
```

Write `case.toml`. This template follows `cases/analyses/SHTB_PC/case.toml`:

```toml
kind = "analysis"
bars = "../../identifications/my_shtb_cal"   # the identification to reuse

loading = "tension"
data = "specimen_record.txt"
units = "N"
interface = "specimen"          # bars do not touch; only the reporting depends on it

holder_length = 106.0           # mm, bar face -> specimen, same on both bars.
                                # 0 when the specimen is fixed straight to the bars.

gauges = [935.0, 119.0, 119.0, 1199.0]  # tape, reported as a second position set

[columns]
time = 0
"in-0"  = 1
"in-1"  = 2
"out-0" = 3
"out-1" = 4

[input_bar]
L_input = 2760.0
diameter = 16.0

[output_bar]
L_output = 2777.0
diameter = 16.0

[specimen]
length = 8.0                    # recorded for reference; the force solve does not use it

[trim]
threshold = 0.05
lead = 900.0                    # keep most of the pre-trigger
baseline = true

[null]
window = 0.75
tol = 2.5e-3
```

The following come from `bars` and are **not** read from this file: c0, the
identified positions, α(f) and c_p(f). Re-identify whenever a gauge is re-bonded,
a bar is swapped or a bar is shortened. The calibration belongs to the bars as
they were on the day of the calibration shot.

**`holder_length`.** A specimen held in holders screwed onto the bar faces is
not loaded by the face force alone. Part of that force accelerates the holder,
which on `SHTB_PC` shows up as an 8.5 kN spike that the ~2 kN specimen never
carries. With `holder_length` set, the force is reported at the
holder/specimen plane, `x = −holder_length`. The holder is treated as more of
the same bar.

- A plain bar-material holder: start from the drawing length.
- A stepped holder: tune the length on force equilibrium (A7). `SHTB_PC` uses
  106 mm against a drawing length of 100 mm.

## A7. Run the analysis

```bash
python3 reconstruct_interface.py cases/analyses/my_specimen
```

It writes:

- `interface_force.png`
- `interface_force.dat`, with columns `time [µs, source-file base]`, `F_in`,
  `F_out` at the specimen plane. The header records c0, the positions, the
  plane and eta.

**What to look at:**

- **Force equilibrium**, `|F_in − F_out| / peak`, in the printout and in the
  bottom-left panel. This is the check on an SHTB specimen shot.
  - `SHTB_PC` reads 0.093 mean at the holder/specimen plane, against 0.40 at the
    bare faces.
  - Early in the loading a mismatch is physics: the specimen has not reached
    equilibrium yet.
  - A persistent mismatch is model error: a holder, a stale calibration or the
    wrong plane.
- **Two position sets**, identified and tape. The printout gives the difference
  in peak force and in L2. A few percent from a common offset is expected.
- **Free-end null**, on the output bar. With a calibration carried over from
  another day, this is where a stale calibration shows first.
- **Causality `n/a`** is normal on a specimen shot. The echo edge is too smooth
  to time, and the script says so rather than inventing a window.

**Sliders.** In the interactive window each bar has a slider that moves its
reconstruction plane:

- 0 is the holder/specimen interface;
- `+holder_length` is the bar face;
- negative values go past the interface, which is an extrapolation.

Use them to see how sensitive equilibrium is to the plane. To make a better
value permanent, change `holder_length`. The sliders never change the file.

**Comparison run:**

```bash
python3 reconstruct_interface.py cases/analyses/my_specimen --no-attenuation --no-dispersion
```

This shows what α(f) and c_p(f) contribute. It writes
`interface_force_lossless.*` next to the default outputs. With `--no-dispersion`
alone, it overwrites the default files.

## A8. What you have now, and what is not there yet

- **You have** `F_in(t)` and `F_out(t)` at the specimen, in the record's units.
- **Not there yet** is the step from those forces to specimen stress, strain
  and strain rate for a *measured* shot. `simulation_code/reduce_specimen.py` does that
  step only for simulation folders. For now, divide `F_out` by the specimen
  cross-section for stress. Strain needs the particle velocities at the two
  faces, which this pipeline does not yet export for measured shots.

---

# Part B — Direct-impact pressure bar

The rig: the input bar **is** the striker and carries no gauge. It is fired
straight at the output bar, or at a specimen on the output bar's face. The
output bar carries two gauges, `out-0` and `out-1`, and both bars have free far
ends. `x = 0` is the output bar's struck face.

This rig is easier to identify than the SHTB:

- Once the bars part, each rings on its own round trip 2L/c.
- That round trip is visible at every gauge.
- The length that sets the scale is simply **the bar's own length**, measured on
  the bench.
- Gauge positions come out *proportional* to that length, so a 2 mm tape error
  on a 1 m bar moves them by only about 0.2 %.

## B1. Fire the calibration shot

- Fire the striker directly onto the output bar's face, with no specimen.
- Record long enough for the gauges to see the **round trip 2L/c** on the output
  bar, plus margin. On the 1027 mm polycarbonate bar that is 1.46 ms, and a 2 ms
  record is enough.
- Keep a quiet pre-trigger. If the trigger fires late on a slow rise, see B6.

## B2. Measure

| what | key |
|---|---|
| output bar length, face to free end, measured before gluing gauges | `L_free_out_ref` and `[bar] length` |
| input bar (striker) length | `L_free_in_ref`, used only to report the input bar's own c from the pulse length |
| tape positions of the two gauges from the face | `gauges` |
| diameter | `[bar] diameter` |
| density (optional) | `[bar] rho`, used only to report E = ρc², never for the force |

## B3. Create the identification folder

```bash
mkdir cases/identifications/my_bar_cal
cp /path/to/record.txt cases/identifications/my_bar_cal/
```

This template follows `cases/identifications/pc_bar/case.toml`:

```toml
kind = "identification"
method = "compression"           # -> identify_bar_compression.py

loading = "compression"          # compression POSITIVE in this record
data = "record.txt"
gauges = [118.0, 489.0]          # TAPE, never shown to the identification

L_free_out_ref = 1027.0          # THE measured length: the output bar
L_free_ref_tol = 2.0
L_free_in_ref = 2415.0           # striker length, for its c only

[columns]                        # a single [bar] table: names are out-*
time = 0
"out-0" = 1                      # nearer the face
"out-1" = 2

[bar]
length = 1027.0
diameter = 16.7
rho = 1.2e-6                     # optional [kg/mm^3]

[trim]
threshold = 0.05
lead = 50.0
baseline = true

[attenuation]                    # needed for a polymer bar; harmless on metal
f_lo = 2.0
f_hi = 50.0
snr = 0.01

[null]
window = 0.75
tol = 6.0e-2                     # the floor on a viscoelastic bar is higher; measure yours
```

## B4. Run the identification

```bash
python3 identify_bar_compression.py cases/identifications/my_bar_cal
```

To test how sensitive the result is to the bench length, override it with
`--l-out-ref 1029`. The script writes `bar_identified.npz` and
`bar_identification.png`.

**Reading the printout:**

1. **Edges per gauge.**
   - `f1` is the arrival, `f2` the free-end echo and `f3` the round trip back
     at the face.
   - The `2L/c` column must agree across gauges. A spread of about 1 µs is good.
   - Two edges that every gauge shares are recognised and excluded: the striker
     unloading and the bars parting.
2. **Wave speed.** One c per bar, from its own length. The input bar's c is
   inferred from the striker pulse length.
3. **Gauge positions.** The identified `x` is shown next to the tape value.
   - A **common** offset is harmless. On `pc_bar` both gauges are about 9 mm
     off, because the face is not an ideal free end while the striker is still
     in contact.
   - The **spacing D** is what matters. It agrees with the tape to 0.9 mm
     there.
4. **Attenuation.** A polymer bar needs α(f). Look at "far gauge predicted from
   the near one", which should drop well below the lossless value (0.24 against
   0.42 on `pc_bar`).
5. **Free-end null.** On a direct impact the floor is higher than on the SHTB,
   because the record starts from a velocity step. Set `[null] tol` from a
   measured floor, not a guess. `pc_bar` passes at 3.5e-2, against 8.2e-2 with a
   lossless bar.

## B5. Optional: check the calibration shot itself

```bash
python3 reconstruct_interface.py cases/identifications/my_bar_cal
```

Two bare bars in contact have to obey boundary conditions that need no ground
truth:

- the contact force is never tensile;
- it drops to zero once the bars part;
- `M` is zero at the face until the echo can return.

The script scores all three.

## B6. Create the analysis folder for a specimen shot

This template follows `cases/analyses/pc_specimen/case.toml`:

```toml
kind = "analysis"
bars = "../../identifications/my_bar_cal"

loading = "compression"
data = "specimen_record.txt"
gauges = [118.0, 489.0]          # tape, the second position set
L_free_out_ref = 1027.0
L_free_in_ref = 2415.0
L_free_ref_tol = 2.0
interface = "specimen"           # a specimen sits between striker and output bar

[columns]
time = 0
"out-0" = 1
"out-1" = 2

[bar]
length = 1027.0
diameter = 16.7
rho = 1.2e-6

[trim]
# Default: arrival detection, as in B3. For a record whose trigger fires LATE
# (a slow rise before t = 0), take the baseline before the rise and keep the
# whole record instead:
baseline_before = -1200.0        # us, before the slow rise
start = -1638.0                  # us, i.e. keep everything

[null]
window = 0.75
tol = 6.0e-2
```

`[trim]` is the part to get right on this rig. On `pc_specimen`, cutting the
record at 500 µs "to start at the interesting part" left a standing 0.13 kN
load in the bar, and the free-end null went from 3.1e-2 to 1.9e-1. Start where
the record is quiet.

## B7. Run the analysis

```bash
python3 reconstruct_interface.py cases/analyses/my_specimen
python3 time_shift_output_gauges_to_interface.py cases/analyses/my_specimen   # optional
```

It writes:

- `interface_force.png`
- `interface_force.dat`, with columns `time [µs, source-file base]` and
  `F_out`: the force at the output bar / specimen interface.

**What to look at:**

- **Free-end null.** It is the main check here, because there is no second bar
  to test equilibrium against. It should stay close to the calibration's value:
  `pc_specimen` reads 3.1e-2 against 3.5e-2 on `pc_bar`. A clear rise means a
  stale calibration or a bad trim.
- **Tensile excursion** (`F < 0`). The specimen is pushed and never pulled, so a
  tensile force is model error with a known sign.
- **Two position sets.** The identified and tape positions should give almost
  the same force: 2 % L2 on `pc_specimen`.
- **Causality `n/a`** is normal, because the specimen smooths the echo edge.
- **`time_shift_output_gauges_to_interface.py`** shifts each gauge to the face on its own,
  which is what a classical single-gauge reduction does. It overlays that on the
  two-gauge separation. Where they agree, one gauge would have been enough.
  Where they peel apart, the single-gauge answer is wrong by the whole of the
  wave it neglected.

## B8. What you have now

- **You have** `F_out(t)` on the specimen's output side.
- **You do not have** the input-side force. The striker has no gauges, so on
  this rig force equilibrium across the specimen cannot be checked from the
  record.
- **Not there yet** is stress-strain reduction of a measured shot, as in A8.

---

## Troubleshooting

| message or symptom | cause and fix |
|---|---|
| `... bar_identified.npz ... run identify_bar_<method>.py <dir>` | The analysis's `bars` folder has not been identified yet. Run the command it names. |
| `... has kind = "..."` | A folder is in the wrong place, or it was passed to the wrong script. `kind` must match the parent directory. |
| `N tape positions but M gauge channels` | `gauges` and `[columns]` disagree in length. They must be in the same order. |
| `gauge 'x' must be named "in-*" or "out-*"` | Two-bar cases split the channels by name prefix. |
| `time column is not uniformly sampled` | Resample the record; the FFT-based separation requires a uniform time step. |
| echo missing from the edge table | The record is too short (A1, B1) or clipped (see `clip_onset`). |
| free-end null FAIL | The transit times are wrong: coupler, bar length, a re-bonded gauge or a stale calibration. It is not a tape-scale error, because the test cannot see one. |
| slow rise before t = 0, null much worse than on the calibration | Late trigger. Use `baseline_before` and `start` (B6). |

**eta.** The default is `eta = 1.0` /ms in `defaults.toml`. The force is
insensitive to it above about 0.5 /ms. Do not lower it to make the force look
cleaner: under-regularising ruins the integrated strain long before the stress
shows it. See "Choosing eta" in `README.md`.
