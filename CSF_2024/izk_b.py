#########################################################################
###### Implementation of UKF to determine the coupling coeff. ...   #####
#########          Izhikevich neurons - electric coupling      ##########
############### 	   by      Giulio Tirabassi         #################
###############  comments      Raul P. Aristides        #################
#########################################################################

import numpy as np


class IZKSimulator:
    def __init__(
        self, dt, coupling, process_noise, measurement_noise, A=None, seed=None
    ):
        self.A = A if A is not None else 1 - np.eye(3)
        self.dt = dt
        self.sdt = np.sqrt(dt)
        self.sigma = process_noise
        self.rho = measurement_noise
        self.K = coupling
        self._x = None
        self._x_premeasurement = None
        self.dimension = int(self.A.shape[0] * 2)
        self.random = np.random.RandomState(seed)

    def simulate_izhikevich(self, x0, n_steps):
        self._x = []
        x = [np.array(x0)]
        for _ in range(n_steps):
            new_deterministic_x = self.RK4(x[-1], self.Izkv2 , self.dt, KA=None)
            new_x = new_deterministic_x + self.random.normal(
            size=self.dimension, scale=self.dt * self.sigma)
            x.append(new_x)
            self._x.append(new_deterministic_x)
        x = np.stack(x)
        self._x = np.stack(self._x)
        self._x_premeasurement = x.copy()
        x += self.random.normal(scale=self.rho, size=x.size).reshape(x.shape)
        return x

    def Izkv2(self,x, KA = None, alpha = 0.2, B = 2, I = -99):
        if KA is None:
            KA = self.K * self.A
        xyz = x.copy()
        v, u = xyz[::2], xyz[1::2]
        cpl = -(v[:, np.newaxis] - v)
        lv = 0.04 * v **2 + 5 * v + 140 - u + I + (KA * cpl).sum(axis=1)
        lu = alpha * ( B*v - u)
        xyz[::2], xyz[1::2] = lv, lu
        return xyz


    def RK4(self,x,f,dt,KA = None,thre = 30, C = -56, D = -16):
        if KA is None:
            KA = self.K * self.A
        xvc = x.copy()
        k1x = dt * f(xvc,KA)
        k2x = dt * f(xvc +  k1x / 2.0, KA)
        k3x = dt * f(xvc +  k2x / 2.0, KA)
        k4x = dt * f(xvc +  k3x, KA)
        xvc += ( k1x + 2.0 * k2x + 2.0 * k3x + k4x ) / 6.0
        v, u = xvc[::2], xvc[1::2]
        rst = (v >= thre).nonzero()[0]
        v[rst] = C + np.zeros(rst.shape[0])
        u[rst] = u[rst] + D
        xvc[::2], xvc[1::2] = v, u
        return xvc

    def extended_step_forward(self, x, dt):
        KA = self.build_coupling_from_extended_state(x)
        params = x[self.dimension:]
        x = x[: self.dimension]
        return np.concatenate([self.RK4(x, self.Izkv2, dt, KA), params])

    def build_coupling_from_extended_state(self, x):
        KA = np.zeros(self.A.shape)
        KA[np.triu_indices(self.A.shape[0], k=1)] = x[self.dimension :] #**
        KA += KA.T #then sum the lower diagonal block
        return KA
