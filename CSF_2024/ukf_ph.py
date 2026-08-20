#############################################################
##    Implementation of UKF to determine the coupling      ##
##          of a network of Izhikevich neurons             ##
##                 from their spiking data                 ##
##############  by  Raul de Palma Aristides   ###############
#############################################################
#  Reference : https://doi.org/10.1016/j.chaos.2024.114837  #
#############################################################

import numpy as np
import pickle
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
import itertools
import  concurrent.futures 
from izk_b import IZKSimulator
from kuramoto import KuramotoSimulator as KS
import matplotlib.pyplot as plt
np.random.seed(0)


#Parameters for simulation Izhikevich neurons
N = 3
Adj = np.array([[1,0,1],[1,1,1]])# # a line of three oscillators (o-o-o) and all to all.

N_STEPS_SIMULATION = 50000# 85000 more or less 72 spikes
N_STEPS_TRANSIENT = int(N_STEPS_SIMULATION) 
DT = 0.01
PROCESS_NOISE = 0.05
MEASUREMENT_NOISE = 0.05
K = 0.1
######################################################################
######################################################################
def kop(X):
	'''
	Function to estimate the Kuramoto order parameter of the simulated network.
	'''
	m = X.shape[0]
	T = X.shape[1]
	n = X.shape[2]
	R = np.zeros(m)	
	for j in range(m):
		P = 0
		Q = 0
		for i in range(n):
			P+= np.sin(X[j,:,i])	
			Q+= np.cos(X[j,:,i])
		R[j] = np.mean( np.sqrt(P**2 + Q**2) / n)
	return R

######################################################################
######################################################################
def calculatespks(entry):
	'''
	Simulates a network of coupled Izhikevich neurons for a given topology/coupling,
	then converts each neuron's spike times into a continuous phase (theta2) via
	linear interpolation between spikes. Edges are trimmed since phase there isn't
	reliable at boundaries (see note below). 
	These phases are the measurements later fed into the UKF.
	'''
	N = int(len(entry[2])/2)
	Ads = np.zeros((N,N))
	Ads[np.triu_indices(n=N,k=1)] = Adj[entry[0]]
	Ads += Ads.T

	simulator = IZKSimulator(
		dt=DT,
		process_noise=PROCESS_NOISE,
		measurement_noise=MEASUREMENT_NOISE,
		coupling=entry[1],
		A=Ads, # a line of three k.o. (o-o-o)
		seed=1,
	)

	x0 = entry[2]#initial conditions
	xt = simulator.simulate_izhikevich(x0=x0, n_steps=N_STEPS_TRANSIENT)#transient
	x = simulator.simulate_izhikevich(x0=xt[-1], n_steps=N_STEPS_SIMULATION)#final run
	x = x[:,::2]#downsampling

	n, N = x.shape
	spks = np.zeros((n,N))

	V = x
	data_spks = []
	dict_spks = {}
	dict_spks["topology"] = entry[0]
	dict_spks["c.strength"] = entry[1]
	
	for j in range(N):#from voltages to phases
		V = x
		V = V[:,j]

		spiketimes  = []
		spiketimes.append(0)

		lst = np.where(V > 26)[0]
		lst = [i for i in lst if i < (n-1000)]

		for i in lst:
			if((V[i] > V[i+1]) & (V[i] > V[i-1])):
				spiketimes.append(i)

		data_spks.append(spiketimes)
		theta2 = np.zeros(n)
		k = 0
		for i in range(n):
			theta2[i] = 2*np.pi*((i - spiketimes[k])/(spiketimes[k+1] - spiketimes[k])) + 2*np.pi*k
			if (i == spiketimes[k+1]):		
				if(i != max(spiketimes)):			
					k+=1

		spks[:,j] = theta2

	dict_spks["spiketimes"] = data_spks
	'''	
	Finally, we need some trimming of the 'data_spks'.
	First, spiketimes[0]=0 is a placeholder, not a real spike, so phase before the first true spike is miscalibrated.
	Second, spikes are only searched for i < n-1000, so the tail phase is extrapolated from the last known inter-spike interval, not measured.
	Keep in mind that 5000 is a conservative fixed margin (not tied to actual spike rate),
	so it may over- or even under-trim depending on topology/coupling strength.
	'''	
	return spks[5000:(n-5000),:],dict_spks

######################################################################
######################################################################
def calculateukf(entries):
	'''	
	Usual UKF routine, here we use the phases from the 'calculatespks' function. We keep the average and standard deviation of the estimated adjacency matrix entries.
	'''	
	DT = 0.01
	PROCESS_NOISE = 0.01
	MEASUREMENT_NOISE = 0.02
	K = 0.1
	N_IT_KF = 2#numbers of times that the whole code is executed
	K0 = 0.1
	P0 = 0.05

	x = np.array(entries[1])
	
	x0 = x[0,:]
	o_mean = np.mean(x,axis = 1)
	o_mean = (o_mean[-1] - o_mean[0]) / x.shape[0]

	simulator = KS(
		dt=DT,
		process_noise=PROCESS_NOISE,
		measurement_noise=MEASUREMENT_NOISE,
		coupling=K,
		omega= np.zeros(N)+o_mean, 
		A= np.zeros((N,N)), 
		seed=1,
	)
	# Define Unscented Kalman Filter
	z_dim = x.shape[1] # = 3
	x_dim = z_dim + (z_dim * (z_dim - 1)) // 2 # = 6 x's and \dot{x}'s ?

	points = MerweScaledSigmaPoints(x_dim, alpha=0.001, beta=2, kappa=3 - x_dim)
	kalman_filter = UnscentedKalmanFilter(
		dim_x=x_dim,
		dim_z=z_dim,
		dt=simulator.dt,
		fx=simulator.extended_step_forward,
		hx=lambda x: x[0:z_dim], # measuring the oscillators
		points=points,
	)
	kalman_filter.R *= simulator.rho
	kalman_filter.Q = np.diag(
		[simulator.sdt * simulator.sigma] * z_dim + [1e-10] * (x_dim - z_dim)
	)
	kalman_filter.P *= P0

	# Run filter
	all_means_coupling = []
	all_vars_coupling = []
	all_std_coupling = []
	params0 = np.zeros(x_dim - z_dim) + K0
	for _ in range(N_IT_KF):
		kalman_filter.x = np.hstack([x0, params0]) #each time get new values of KA
		mean_x, cov_x = kalman_filter.batch_filter(x)
		var_x = cov_x[:, np.arange(x_dim), np.arange(x_dim)] #get diagonals
		all_means_coupling.append(mean_x[:, z_dim:])
		all_vars_coupling.append(var_x[:, z_dim:])
		print(mean_x[:, z_dim:].shape, var_x[:, z_dim:].shape)
		params0 = mean_x[-1, z_dim:]

	all_means_coupling = np.vstack(all_means_coupling)
	all_std_coupling = np.vstack(all_vars_coupling) ** 0.5

	return all_means_coupling, all_std_coupling

######################################################################
######################################################################
######################################################################
######################################################################
sel = [0]#We'll use just the o-o-o network in this example. Otherwise: sel = Adj.shape[0] 
lkas = 2
kas = np.linspace(0,0.075,lkas) #coupling strengths used in this example, just two.

master_scores = []
master_kop = []
master_spk = []

for _ in range(1):#Number of trials, you can run many of them and them average the estimations to get a better result.
	spkk = []
	data_sk = []
	lista = []
	for i in sel:
		for j in kas:
			lista.append([i,j])
	for i in lista:
		ic = 5 - 10*np.random.rand(2*N) + np.array(N*[-56.25,-112.5]) #Initial conditions around the attractor.
		i.append(ic)

	with concurrent.futures.ProcessPoolExecutor() as executor:
		sns = [i for i in lista] 
		results = executor.map(calculatespks, sns)
		for result in results:
			spkk.append(result[0])
			data_sk.append(result[1])

	spkk = np.array(spkk)

	K_R = kop(spkk).reshape(len(sel),lkas)
	master_kop.append(K_R)
	master_spk.append(data_sk)# See the comment about this data in the plotting.py file.

	means = []
	stand = []
	with concurrent.futures.ProcessPoolExecutor() as executor:
		yns = [[i,spkk[i]] for i in range(spkk.shape[0])] 
		results = executor.map(calculateukf, yns)
		for result in results:
			means.append(result[0])
			stand.append(result[1])

	means = np.array(means)
	stand = np.array(stand)
	master_kop = np.array(master_kop)


np.save("mn4_korder_up.npy",master_kop)
np.save("mn4_means_up.npy", means)
np.save("mn4_stand_up.npy", stand)

file_name = "spikedata.pkl"
open_file = open(file_name,"wb")
pickle.dump(master_spk,open_file) # See the comment about this data in the plotting.py file.
open_file.close()
