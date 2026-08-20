# Coupled Oscillator/Neuron Network Synchronization — Code Repository

This repository contains the simulation and analysis code for two related
papers on synchronization dynamics in small networks of coupled oscillators
and Izhikevich neurons, using an Unscented Kalman Filter (UKF) to estimate
coupling parameters from spike/phase data.

## Repository structure

Each subdirectory is self-contained and has its own README with details on
what the code does, how to run it, and paper-specific notes.

## Papers

Codes and supplementary info of the papers 1:https://doi.org/10.1063/5.0144499 (2023) and 2: https://doi.org/10.1016/j.chaos.2024.114837 (2024)

Resources


- **Paper 1** — *[Parameter and coupling estimation in small networks of Izhikevich’s neurons]*. [https://doi.org/10.1063/5.0144499]
- **Paper 2** — *[Inferring the connectivity of coupled oscillators from event timing analysis]*. [https://doi.org/10.1016/j.chaos.2024.114837]

## Requirements

Both projects rely on filterpy, which can be installed with:

```bash
pip install numpy filterpy matplotlib pandas
```

## Citation
If you use this code, please cite the relevant papers.
