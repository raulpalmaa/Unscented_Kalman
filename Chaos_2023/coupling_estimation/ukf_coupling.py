#############################################################
##    Implementation of UKF to determine the coupling      ##
##          of a network of Izhikevich neurons             ##
##############  by  Raul de Palma Aristides   ###############
##############      and Giulio Tirabassi      ###############
####### Reference : https://doi.org/10.1063/5.0144499 #######
#############################################################

from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from numpy.lib.function_base import _median_dispatcher
from izk_d import IZKSimulator
from matplotlib import rc
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(99)

N_STEPS_SIMULATION = 5000
N_STEPS_TRANSIENT = 5000#int(N_STEPS_SIMULATION/5) 
DT = 0.01	
PROCESS_NOISE = 0.01
MEASUREMENT_NOISE = 0.02
K = 0.05
N_IT_KF = 1 #numbers of times that the whole code is executed
K0 = 0.3 #first estimation of K 
P0 = 0.05

AdjEs = np.array([[[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
 [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]],
 [[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
 [[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0]],
 [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]])

AdjCs = 0.5 * np.array([[[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]],
 [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]],
 [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
 [[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 0, 0]],
 [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]]])

tpgs = ['chain','ring','star','sling','altal']
all_m_l = []
all_s_l = []
xx = []

N = AdjEs[0].shape[0]

for i in [0]:
	all_m_l = []
	all_s_l = []
	for _ in range(1):	# Perform Kuramoto simulations
		simulator = IZKSimulator(
			dt=DT,
			process_noise=PROCESS_NOISE,
			measurement_noise=MEASUREMENT_NOISE,
			coupling=K,
			AE= AdjEs[i],
			AC= AdjCs[i],
			seed=1)

		x0 = np.array([-60.57, -115.09,-60.57, -115.09,-60.57, -115.09,-60.57, -115.09 ]) + 2*np.random.rand(int(2*N))

		#transient:
		xt = simulator.simulate_izhikevich(x0=x0, n_steps=N_STEPS_TRANSIENT)
		#for real now:
		x = simulator.simulate_izhikevich(x0=xt[-1], n_steps=N_STEPS_SIMULATION)
		xx.append(x[:,::2])

#np.save("ukf_d_x_0.05_ex.npy",np.array(xx))

		ttl = simulator.tbd + simulator.lae  # total # of elements to be evaluated 
		print(ttl,'ttl size')
		# Define Unscented Kalman Filter
		z_dim = len(x0)#int(len(x0)/2) # = 3
		x_dim = z_dim + ttl # = 6 + 3

		points = MerweScaledSigmaPoints(x_dim, alpha=0.001, beta=2, kappa=3 - x_dim)
		kalman_filter = UnscentedKalmanFilter(
			dim_x=x_dim,
			dim_z=z_dim,
			dt=simulator.dt,
			fx=simulator.extended_step_forward,
			hx=lambda x: x[0:z_dim], # measuring the oscillators
			points=points,
		)
		kalman_filter.R *= simulator.rho #measurement noise
		kalman_filter.Q = np.diag(
			[simulator.sdt * simulator.sigma] * z_dim + [1e-10] * (x_dim - z_dim)
		) #process noise
		kalman_filter.P *= P0 #covariance estimate vector

		# Run filter
		all_means_coupling = []
		all_vars_coupling = []

		params0 = np.zeros(ttl) + K0#symmetric coupling


		for _ in range(N_IT_KF): #at each loop we start with a better guess 
			kalman_filter.x = np.hstack([x0, params0]) #each time get new values of KA
			mean_x, cov_x = kalman_filter.batch_filter(x) #Performs the UKF filter over the list of measurement in zs.
			var_x = cov_x[:, np.arange(x_dim), np.arange(x_dim)]
			all_means_coupling.append(mean_x[:, len(x0):])
			all_vars_coupling.append(var_x[:, len(x0):])
			params0 = mean_x[-1, len(x0):]

		all_means_coupling = np.vstack(all_means_coupling)
		all_std_coupling = np.vstack(all_vars_coupling) ** 0.5
		all_m_l.append(all_means_coupling)
		all_s_l.append(all_std_coupling)

	all_m_l = np.array(all_m_l)
	all_s_l = np.array(all_s_l)

print(mean_x.shape)
print(all_m_l[0,-1,:])
print(all_m_l[0,-1,:].shape)


AdjE = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
AdjC = 0.5 * np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]])

inferred = all_m_l[0,-1,:]

lae = simulator.lae
extra = simulator.extra

# --- KAE: fully symmetric ---
KAE = np.zeros((N, N))
KAE[np.triu_indices(N, k=1)] = inferred[0:lae]
KAE += KAE.T

# --- KAC: symmetric baseline + asymmetric patch ---
KAC = np.zeros((N, N))
KAC[np.triu_indices(N, k=1)] = inferred[lae:lae + (6)]  # 6 = triu size for AC baseline
KAC += KAC.T

for el in range(extra):
    KAC[simulator.tlist[el]] = inferred[lae + 6 + el]   # sim = your KuramotoSimulator instance

fig, axs = plt.subplots(2,1)

axs[0].axhline(y= K, color="k", ls="--",  alpha = 0.6)
axs[0].axhline(y= 0, color="k", ls="--",  alpha = 0.6)
axs[0].plot(AdjEs[0].flatten()*K,'kD',alpha = 0.5, ms = 8)
axs[0].plot(KAE.flatten(),'bo',markeredgecolor='black')

axs[1].plot(AdjCs[0].flatten()*K,'kD',alpha = 0.5, ms = 8)
axs[1].plot(KAC.flatten(),'ro',markeredgecolor='black')

axs[1].axhline(y= K*0.5, color="k", ls="--", alpha = 0.6)
axs[1].axhline(y= 0, color="k", ls="--",  alpha = 0.6)

axs[0].set_ylabel('Matrix entry')
axs[1].set_ylabel('Matrix entry')
axs[1].set_xlabel('(ij)')

entries = [f"({i}{j})" for i in range(N) for j in range(N)]
axs[1].set_xticks(np.arange(N*N), entries)

plt.show()
