#############################################################
##    Implementation of UKF to determine the parameters    ##
#########          of a Izhikevich neuron          ##########
##############  by  Raul de Palma Aristides #################
##############      and Giulio Tirabassi      ###############
####### Reference : https://doi.org/10.1063/5.0144499 #######
#############################################################

import numpy as np

class IZKSimulator:
    def __init__(
        self, dt, process_noise, measurement_noise, p_a, p_b, Ix, seed=None
    ):
        self.dt = dt
        self.sdt = np.sqrt(dt)
        self.sigma = process_noise
        self.rho = measurement_noise
        self.alpha = p_a
        self.beta = p_b
        self.Ic = Ix
        self._x = None
        self._x_premeasurement = None
        self.dimension =  2
        self.random = np.random.RandomState(seed)

    def simulate_izhikevich(self, x0, n_steps):
        self._x = []
        x = [np.array(x0)]
        I_n = self.Ic * np.ones(n_steps) 
        for cnt in range(n_steps):
            new_deterministic_x = self.RK4(x[-1], self.Izkv , self.dt, I = I_n[cnt])
            new_x = new_deterministic_x + self.random.normal(
            size=self.dimension, scale=self.sdt * self.sigma )
            x.append(new_x)
            self._x.append(new_deterministic_x)
        x = np.stack(x)
        self._x = np.stack(self._x)
        self._x_premeasurement = x.copy()
        x += self.random.normal(scale=self.rho, size=x.size).reshape(x.shape)
        return x    

    def Izkv(self, x, I, aa = None, bb = None):
        if I is None:
            I = self.Ic
        if aa is None:
            aa = self.alpha
        if bb is None:
            bb = self.beta
        xyz = x.copy()
        v, u = xyz[::2], xyz[1::2]
        lv = 0.04 * v **2 + 5 * v + 140 - u + I #+ A * np.sin(omega * t)
        lu = aa * ( bb*v - u)
        xyz[::2], xyz[1::2] = lv, lu
        return xyz


    def RK4(self, x, f, dt, I, aa = None, bb = None, thre = 30, cc = -56, dd = -16):
        if I is None:
            I = self.Ic
        if aa is None:
            aa = self.alpha
        if bb is None:
            bb = self.beta
        xvc = x.copy()
        k1x = dt * f(xvc,I, aa, bb)
        k2x = dt * f(xvc +  k1x / 2.0, I, aa, bb)
        k3x = dt * f(xvc +  k2x / 2.0, I, aa, bb)
        k4x = dt * f(xvc +  k3x, I, aa, bb)
        xvc += ( k1x + 2.0 * k2x + 2.0 * k3x + k4x ) / 6.0
        v, u = xvc[::2], xvc[1::2]
        rst = (v >= thre).nonzero()[0]
        v[rst] = cc + np.zeros(rst.shape[0])
        u[rst] = u[rst] + dd
        xvc[::2], xvc[1::2] = v, u
        return xvc

    def extended_step_forward(self, x, dt):
        aa, bb, I = x[-3:]
        params = x[self.dimension:]
        x = x[: self.dimension]
        return np.concatenate([self.RK4(x, self.Izkv, dt, I, aa, bb), params])

