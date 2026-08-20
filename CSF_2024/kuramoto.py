#########################################################################
###### Implementation of UKF to determine the coupling coeff. ...   #####
########                Kuramoto oscillators class              #########
############### 	   by      Giulio Tirabassi         #################
###############  comments      Raul P. Aristides        #################
#########################################################################

import numpy as np


class KuramotoSimulator:
    def __init__(
        self, omega, dt, coupling, process_noise, measurement_noise, A=None, seed=None
    ):
        self.omega = np.array(omega)
        self.A = A if A is not None else 1 - np.eye(self.omega.size)
        self.dt = dt
        self.sdt = np.sqrt(dt)
        self.sigma = process_noise
        self.rho = measurement_noise
        self.K = coupling
        self._x = None
        self._x_premeasurement = None
        self.dimension = self.omega.size
        self.random = np.random.RandomState(seed)

    def simulate_kuramoto(self, x0, n_steps):
        self._x = []
        x = [np.array(x0)]
        for _ in range(n_steps):
            new_deterministic_x = self.deterministic_step_forward(x[-1], self.dt)
            new_x = new_deterministic_x + self.random.normal(
                size=self.dimension, scale=self.sdt * self.sigma
            )
            x.append(new_x)
            self._x.append(new_deterministic_x)
        x = np.stack(x)
        self._x = np.stack(self._x)
        self._x_premeasurement = x.copy()
        x += self.random.normal(scale=self.rho, size=x.size).reshape(x.shape)
        return x

    def deterministic_step_forward(self, x, dt, KA=None):
        if KA is None:
            KA = self.K * self.A
        deltax = -(x[:, np.newaxis] - x) # A convenient alias for None
        rhs = self.omega + (KA * np.sin(deltax)).sum(axis=1)
        return x + rhs * dt

    def extended_step_forward(self, x, dt):
        KA = self.build_coupling_from_extended_state(x)
        params = x[self.dimension :]
        x = x[: self.dimension]
        return np.concatenate([self.deterministic_step_forward(x, dt, KA=KA), params])

    def build_coupling_from_extended_state(self, x):
        KA = np.zeros(self.A.shape)
        KA[np.triu_indices(self.dimension, k=1)] = x[self.dimension :] #**
        KA += KA.T
        return KA
