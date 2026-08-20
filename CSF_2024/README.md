# Coupled Izhikevich Network — Spike-to-Phase UKF Estimation

This repo simulates a small network of coupled Izhikevich neurons under different
connection topologies and coupling strengths, extracts spike times, converts them
into continuous phase signals, and then uses an Unscented Kalman Filter (UKF) to
estimate the underlying coupling parameters from those phases. It also includes a
raster plot script to visualize the spiking activity.

## What it does

1. **Simulate.** For each topology (adjacency matrix) and coupling strength, a
   network of Izhikevich neurons is simulated: first a transient run to let the
   system settle, then a longer run used for analysis.
2. **Detect spikes.** Each neuron's voltage trace is scanned for local maxima
   above a threshold to find spike times.
3. **Spikes → phase.** Between consecutive spikes, phase is linearly interpolated
   from 0 to 2π, giving a continuous phase signal (`theta2`) per neuron. The very
   start and end of each trace are trimmed, since phase there is not reliable
   (see comments in `calculatespks`).
4. **Order parameter.** The Kuramoto order parameter is computed across neurons
   to quantify synchronization for each topology/coupling combination.
5. **UKF estimation.** The reconstructed phases are fed into an Unscented Kalman
   Filter as noisy measurements, which estimates the effective coupling
   parameters (mean ± standard deviation) driving the network.
6. **Raster plots.** Spike times saved from the simulation can be visualized as
   raster plots, one panel per coupling strength, one row per neuron.

## Repo structure

```
.
├── kuramoto.py          # Izhikevich/Kuramoto simulator used for the UKF step
├── kuramoto_b.py         # Izhikevich/Kuramoto simulator used for spike simulation
├── main.py                # simulation + spike detection + phase extraction + UKF
├── raster_plot.py         # loads saved spike times and plots rasters
├── mn4_scores_up.npy       # saved UKF coupling estimates (generated)
├── mn4_korder_up.npy       # saved Kuramoto order parameters (generated)
└── lol_up.pkl              # saved spike time data per topology/coupling (generated)
```

## Requirements

- Python 3.x
- `numpy`
- `filterpy`
- `matplotlib`
- `pandas`

Install with:

```bash
pip install numpy filterpy matplotlib pandas
```

## Usage

Run the main simulation + estimation pipeline:

```bash
python main.py
```

This will:
- simulate the selected topologies and coupling strengths,
- save spike data to `lol_up.pkl`,
- save UKF coupling estimates to `mn4_scores_up.npy`,
- save order parameters to `mn4_korder_up.npy`.

Then generate raster plots from the saved spikes:

```bash
python raster_plot.py
```

## Notes / known limitations

- The phase-reconstruction trims a fixed margin (5000 samples) from each end of
  the trace, since the first "spike" is a placeholder at t=0 and the last stretch
  is extrapolated rather than measured. This margin is not adaptive to the actual
  spike rate, so it may over- or under-trim depending on topology/coupling.
- The outer simulation loop silently swallows exceptions (`try/except: pass`),
  so failed runs won't raise errors — useful to know if results look incomplete.
- Random initial conditions are not seeded, so runs are not fully reproducible
  even though the simulator itself uses a fixed seed.

## License

Add your license of choice here (MIT, GPL, etc).
