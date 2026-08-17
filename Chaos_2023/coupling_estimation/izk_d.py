#############################################################
##    Implementation of UKF to determine the coupling      ##
##          of a network of Izhikevich neurons             ##
##############  by  Raul de Palma Aristides   ###############
##############      and Giulio Tirabassi      ###############
####### Reference : https://doi.org/10.1063/5.0144499 #######
#############################################################

import numpy as np


class IZKSimulator:
    def __init__(
        self, dt, coupling, process_noise, measurement_noise, AE=None, AC=None, seed=None
    ):
        self.AE = AE if AE is not None else 1 - np.eye(3)
        self.AC = AC if AC is not None else 1 - np.eye(3)
        self.lae = len(self.AE[np.triu_indices(self.AE.shape[0], k=1)]) # total # of elements to be evaluated from AE
        # How many free parameters do we need to estimate AC?
        # Shortcut: assume AC is mostly symmetric, so we only estimate the upper
        # triangle and mirror it (KAC += KAC.T). Any entries where AC breaks
        # symmetry get estimated individually as "extra" params on top of that.
        # (A fully blind setup, with no known structure, would need to estimate
        # all n*(n-1) off-diagonal entries instead.)
        if ((AC[np.triu_indices(self.AC.shape[0], k =1)] == 0).all()):
        # Upper triangle is all zero -> no symmetric structure to exploit,
        # every nonzero entry (all in the lower triangle) needs its own param.
            self.extra = len(np.where(AC[np.tril_indices(self.AC.shape[0], k =-1)] != 0)[0])
            self.alist = np.argwhere(self.AC != 0)
            self.tlist = [tuple(i) for i in self.alist]
            self.tbd = len(self.AC[np.triu_indices(self.AC.shape[0], k=1)]) + self.extra
        else:
        # Some symmetric structure exists; find entries where AC != AC.T
        # (asymmetric couplings) and estimate those individually too.
        	# Each asymmetric pair is counted twice (i,j) and (j,i), hence // 2.
            self.extra = len(np.argwhere((self.AC - self.AC.T) != 0)) // 2 # number of elements to be evaluated beyond the upper block
            self.alist = np.argwhere((self.AC - self.AC.T) != 0)[self.extra:] # indices of A that make it asymmetric
            self.tlist = [tuple(i) for i in self.alist] # getting it ready to put it in build_coupling_from_extended_state
            self.tbd = len(self.AC[np.triu_indices(self.AC.shape[0], k=1)]) + self.extra # total # of elements to be evaluated 
        self.dt = dt
        self.sdt = np.sqrt(dt)
        self.sigma = process_noise
        self.rho = measurement_noise
        self.K = coupling
        self._x = None
        self._x_premeasurement = None
        self.dimension = int(self.AE.shape[0] * 2)
        self.random = np.random.RandomState(seed)

    def simulate_izhikevich(self, x0, n_steps):
        self._x = []
        x = [np.array(x0)]
        for _ in range(n_steps):
            new_deterministic_x = self.RK4(x[-1], self.Izkv2 , self.dt, KAE=None, KAC=None)
            new_x = new_deterministic_x + self.random.normal(
            size=self.dimension, scale=self.sdt * self.sigma)
            x.append(new_x)
            self._x.append(new_deterministic_x)
        x = np.stack(x)
        self._x = np.stack(self._x)
        self._x_premeasurement = x.copy()
        x += self.random.normal(scale=self.rho, size=x.size).reshape(x.shape)
        return x

    def Exp(self, x, nu = 7):
        Exx = (1  / (1 + np.exp(-nu*x)))
        return Exx


    def Izkv2(self, x, KAE = None, KAC = None, alpha = 0.2, B = 2, I = -99, v_s = np.array([0,0,-75,35])): #v_s = 35
        if KAE is None:
            KAE = self.K * self.AE
        if KAC is None:
            KAC = self.K * self.AC
        xyz = x.copy()
        v, u = xyz[::2], xyz[1::2]
        cplE = -(v[:, np.newaxis] - v)
        cplC = - (v - v_s) * (KAC @ (self.Exp(v)))
        lv = 0.04 * v **2 + 5 * v + 140 - u + I + cplC + (KAE * cplE).sum(axis=1)
        lu = alpha * ( B*v - u)
        xyz[::2], xyz[1::2] = lv, lu
        return xyz


    def RK4(self,x, f , dt, KAE = None, KAC = None, thre = 30, C = -56, D = -16):
        if KAE is None:
            KAE = self.K * self.AE
        if KAC is None:
            KAC = self.K * self.AC        
        xvc = x.copy()
        k1x = dt * f(xvc, KAE, KAC)
        k2x = dt * f(xvc +  k1x / 2.0, KAE, KAC)
        k3x = dt * f(xvc +  k2x / 2.0, KAE, KAC)
        k4x = dt * f(xvc +  k3x, KAE, KAC)
        xvc += ( k1x + 2.0 * k2x + 2.0 * k3x + k4x ) / 6.0
        v, u = xvc[::2], xvc[1::2]
        rst = (v >= thre).nonzero()[0]
        v[rst] = C + np.zeros(rst.shape[0])
        u[rst] = u[rst] + D
        xvc[::2], xvc[1::2] = v, u
        return xvc

    def extended_step_forward(self, x, dt):
        KAE, KAC = self.build_coupling_from_extended_state(x)
        params = x[self.dimension:]
        x = x[: self.dimension]
        return np.concatenate([self.RK4(x, self.Izkv2, dt, KAE, KAC), params])

    def build_coupling_from_extended_state(self, x):
        KAE = np.zeros(self.AE.shape)
        KAC = np.zeros(self.AC.shape)
        KAC[np.triu_indices(self.AC.shape[0], k=1)] = x[self.dimension + self.lae : -self.extra] #**
        KAC += KAC.T #then sum the lower diagonal block
        for el in range(self.extra):
            KAC[self.tlist[el]] = x[-self.extra + el] #we stopped here!
        KAE[np.triu_indices(self.AE.shape[0], k=1)] = x[self.dimension : self.dimension + self.lae] #**
        KAE += KAE.T #then sum the lower diagonal block
        return KAE, KAC

'''
def RK4(x, f , dt, KAE = None, KAC = None, thre = 30, C = -56, D = -16):
    if KAE is None:
        KAE = K * AE
    if KAC is None:
        KAC = K * AC        
    xvc = x.copy()
    k1x = dt * f(xvc, KAE, KAC)
    k2x = dt * f(xvc +  k1x / 2.0, KAE, KAC)
    k3x = dt * f(xvc +  k2x / 2.0, KAE, KAC)
    k4x = dt * f(xvc +  k3x, KAE, KAC)
    xvc += ( k1x + 2.0 * k2x + 2.0 * k3x + k4x ) / 6.0
    v, u = xvc[::2], xvc[1::2]
    rst = (v >= thre).nonzero()[0]
    v[rst] = C + np.zeros(rst.shape[0])
    u[rst] = u[rst] + D
    xvc[::2], xvc[1::2] = v, u
    return xvc


def Izkv2(x, KAE = None, KAC = None, alpha = 0.2, B = 2, I = -99, v_s = 0):
    if KAE is None:
        KAE = K * AE
    if KAC is None:
        KAC = K * AC
    xyz = x.copy()
    v, u = xyz[::2], xyz[1::2]
    cplE = -(v[:, np.newaxis] - v)
    cplC = - (v - v_s) * (KAC @ (self.Exp(v)))
    lv = 0.04 * v **2 + 5 * v + 140 - u + I + cplC + (KAE * cplE).sum(axis=1)
    lu = alpha * ( B*v - u)
    xyz[::2], xyz[1::2] = lv, lu
    return xyz


'''
