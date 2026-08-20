# Coupled Oscillator/Neuron Network Synchronization — Code Repository

This repository contains the simulation and analysis code for two related
papers on synchronization dynamics in small networks of coupled oscillators
and Izhikevich neurons. Both use an Unscented Kalman Filter (UKF) to estimate
system parameters — particularly coupling strengths — from data, using either
full time series or spike timing alone

## Repository structure

Each subdirectory is self-contained and has its own README with details on
what the code does, how to run it, and paper-specific notes.

- **Chaos 2023** — *Parameter and coupling estimation in small networks of Izhikevich’s neurons* - https://doi.org/10.1063/5.0144499
- **CSF 2024** — *Inferring the connectivity of coupled oscillators from event timing analysis* - https://doi.org/10.1016/j.chaos.2024.114837

## Requirements

Both projects rely on filterpy, which can be installed with:

```bash
pip install numpy filterpy 
```
Read the documentation: https://filterpy.readthedocs.io/en/latest/

## Citation
If you use this code, please cite the relevant papers.
