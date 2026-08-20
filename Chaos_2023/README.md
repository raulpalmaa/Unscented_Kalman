# Estimating coupling from spiking of neurons with the UKF

This repo contains code to simulate: 

1: A single Izhikevich neuron and then use an Unscented Kalman Filter (UKF) to
estimate the underlying parameters of the model. 
2: A small network of coupled Izhikevich neurons under different
connection topologies and coupling strengths, then uses an Unscented Kalman Filter (UKF) to
estimate the underlying coupling parameters (the connectivity of the network). 

It also includes a plot script to visualize the results.

Together they form a set of minimal scripts that encode the main idea presented in 
*Parameter and coupling estimation in small networks of Izhikevich’s neurons RP Aristides, AJ Pons, HA Cerdeira, C Masoller, G Tirabassi
Chaos: An Interdisciplinary Journal of Nonlinear Science 33 (4), 043123*.

## Repo structure
```
.
├── coupling_estimation     # Codes for coupling parameter estimation, small motifs.
|  ├── izk_d.py          # Defines the Izhikevich model (network) used to generate the data and later encode it with the UKF step. 
|  └── ukf_coupling.py   # Main script: simulation + UKF + plot the results
├── parameter_estimation    # Codes for internal parameter estimation, single node/neuron.
|  ├── izk_isob.npy      # Defines the Izhikevich model (single-node) used to generate the data and later encode it with the UKF step.
|  └── ukf_par.pkl       # Main script: simulation + UKF + plot the results
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
