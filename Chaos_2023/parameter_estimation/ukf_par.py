#############################################################
##    Implementation of UKF to determine the parameters    ##
#########          of a Izhikevich neuron          ##########
##############  by  Raul de Palma Aristides #################
##############      and Giulio Tirabassi      ###############
#############################################################
# Reference 1: https://doi.org/10.1063/5.0144499 
##Reference 2: https://repositorio.unesp.br/entities/publication/f2a78edd-5837-4129-8370-0d70c818bab7
#############################################################

from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from numpy.lib.function_base import _median_dispatcher
from izk_isob import IZKSimulator
from matplotlib import rc
import numpy as np
import matplotlib.pyplot as plt


sam = 1
N_STEPS_SIMULATION = sam * 50000
N_STEPS_TRANSIENT = int(N_STEPS_SIMULATION/5)
DT = 0.01

PROCESS_NOISE = 0.025
MEASUREMENT_NOISE = 0.5
P0 = 0.08
N_IT_KF = 1 #numbers of times that the whole code is executed

pa = 0.2
pb = 2
pc = -56
pd = -16
I = -99

I0 = -90
a0 = 0.25
b0 = 1.8
# Perform Kuramoto simulations
simulator = IZKSimulator(
    dt=DT,
    process_noise=PROCESS_NOISE,
    measurement_noise=MEASUREMENT_NOISE,
    p_a = pa,
    p_b = pb,
    Ix = I,
    seed=1,
)
x0 = np.array([-60.57, -115.09]) + 10*np.random.rand(2)

# Transient:
xt = simulator.simulate_izhikevich(x0=x0, n_steps=N_STEPS_TRANSIENT)
# for real now:
x = simulator.simulate_izhikevich(x0=xt[-1], n_steps=N_STEPS_SIMULATION)

# Define Unscented Kalman Filter
ndp =  3
z_dim = len(x0)#int(len(x0)/2) # = 3
x_dim = z_dim + ndp # = 6 + 3

points = MerweScaledSigmaPoints(x_dim, alpha=0.001, beta=2, kappa=3 - x_dim)
kalman_filter = UnscentedKalmanFilter(
    dim_x=x_dim,
    dim_z=z_dim,
    dt= simulator.dt,
    fx=simulator.extended_step_forward,
    hx=lambda x: x[0:z_dim], # measuring the oscillators
    points=points,
)
kalman_filter.R *= simulator.rho #measurement noise
kalman_filter.Q = np.diag(
    [simulator.sdt * simulator.sigma] * z_dim + [1e-10] * (x_dim - z_dim)
) #process noise
kalman_filter.P *= 3*P0 #covariance estimate vector

# Run filter
all_means = []
all_vars = []

params0 = np.zeros(ndp) 
params0[-1] = I0
params0[-2] = b0
params0[-2] = b0

x0 = xt[-1]
x0 = np.array([-60.57, -115.09]) 
for _ in range(N_IT_KF):
    kalman_filter.x = np.hstack([x0, params0]) 
    mean_x, cov_x = kalman_filter.batch_filter(x[::sam,:])
    var_x = cov_x[:, np.arange(x_dim), np.arange(x_dim)]
    all_means.append(mean_x[:, len(x0):])
    all_vars.append(var_x[:, len(x0):])
    params0 = mean_x[-1, len(x0):]

all_means = np.vstack(all_means)
all_std = np.vstack(all_vars) ** 0.5

poste = mean_x[:,:z_dim]
prior = x[:,:z_dim]

fig, axs = plt.subplots(3,1)

axs[0].axhline(y= pa, color="k", ls="--",  alpha = 0.6)
axs[0].axhline(y= pb, color="k", ls="--",  alpha = 0.6)


for i in range(all_means.shape[1]):
    time = np.arange(all_means.shape[0]) * simulator.dt
    axs[0].plot(time, all_means[:, i])
    axs[0].fill_between(
        time,
        all_means[:, i] - all_std[:, i],
        all_means[:, i] + all_std[:, i],
        alpha=0.4,
    )

axs[0].axhline(y= pa, color="k", ls="--",  alpha = 0.6)
mmin = np.min(all_means[:, :2] - all_std[:, :2])
mmax = np.max(all_means[:, :2] + all_std[:, :2])
axs[0].set_ylim(mmin, mmax)


axs[1].plot(time, all_means[:, 2])
axs[1].fill_between(
    time,
    all_means[:, 2] - all_std[:, 2],
    all_means[:, 2] + all_std[:, 2],
    alpha=0.4,
)
axs[1].axhline(y= I, color="k", ls="--",  alpha = 0.6)

axs[2].plot(time,x[:,0],"r",alpha = 0.8)
axs[2].plot(time,mean_x[:,0],"b",alpha = 0.8)

axs[0].set_ylabel('a, b (a.u.)')
axs[1].set_ylabel('I (a.u.)')
axs[2].set_xlabel('time (a.u.)')
axs[2].set_ylabel('x (a.u.)')

plt.show()
