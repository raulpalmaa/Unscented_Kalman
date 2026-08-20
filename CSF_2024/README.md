# Estimating coupling from spiking of neurons with the UKF

This repo simulates a small network of coupled Izhikevich neurons under different
connection topologies and coupling strengths, extracts spike times, converts them
into continuous phase signals, and then uses an Unscented Kalman Filter (UKF) to
estimate the underlying coupling parameters from those phases. 
It also includes a raster plot script to visualize the spiking activity.
It is a set of minimal scripts that encode the main idea presented in *Inferring the connectivity of coupled oscillators from event timing analysis, RP Aristides, HA Cerdeira, C Masoller, G Tirabassi - Chaos, Solitons & Fractals 182, 114837*.

It builds on our previous work (see the Chaos2023 directory).

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
├── izk_b.py           # Izhikevich simulator used to generate the spikes 
├── kuramoto.py        # Kuramoto simulator used for the UKF step
├── ukf_ph.py          # Main script.
├── plotting.py        # 1: loads and plots the estimated adjacency matrix entries. 2: loads saved spike times and plots rasters

├── kuramoto_b.py         # Izhikevich/Kuramoto simulator used for spike simulation
├── main.py                # simulation + spike detection + phase extraction + UKF
├── raster_plot.py         # loads saved spike times and plots rasters
├── mn4_scores_up.npy       # saved UKF coupling estimates (generated)
├── mn4_korder_up.npy       # saved Kuramoto order parameters (generated)
└── lol_up.pkl              # saved spike time data per topology/coupling (generated)
```

## Usage

Run the main simulation + estimation pipeline:

```bash
python main.py
```

This will:
- simulate the selected topologies and coupling strengths,
- save spike data to `spikedata.pkl`,
- save UKF coupling estimates to `mn4_scores_up.npy`,
- save order parameters to `mn4_korder_up.npy`.

Then plot everything with:

```bash
python plotting.py
```

## Notes

- The phase-reconstruction trims a fixed margin (5000 samples) from each end of
  the trace, since the first "spike" is a placeholder at t=0 and the last stretch
  is extrapolated rather than measured. This margin is not adaptive to the actual
  spike rate, so it may over- or under-trim depending on topology/coupling.

Add your license of choice here (MIT, GPL, etc).
