import numpy as np
import cupy as cp
import copy
from .conv import *

class iSeq:
    
    W = None
    H = None
    V = None
    U = None
    max_iter = None
    tolerance = None
    z_th = None
    k = None
    l = None
    cost = None
    Vmax = None
    Wlim = None
    Hlim = None
    
    def __init__(self, V, k, l, max_iter=100, z_th=0.001, tolerance=1e-7, Hlim=3, Wlim=0.1, random_seed=None):
        
        cp.random.seed(random_seed)
        self.V = V
        n, t = V.shape
        self.Vmax = self.V.max()
        self.W = cp.random.random(size=(n, k, l))
        self.W /= self.W.sum(axis=2).sum(axis=0).reshape(1, -1, 1)
        self.H = cp.random.random(size=(k, t+l-1))
        self.Hlim = Hlim
        self.Wlim = Wlim
        self.max_iter = int(max_iter)
        self.tolerance = tolerance
        self.z_th = z_th
        self.k = k
        self.l = l
        self.cost = []

        self._scale_WH()

    def _scale_WH(self):
        w_sum = self.W.sum(axis=2).sum(axis=0)
        self.H *= w_sum[:, cp.newaxis]
        self.W /= w_sum[cp.newaxis, :, cp.newaxis]
        n = self.W.shape
        while (self.W > self.Wlim).sum() > 0:
            index = self.W >= self.Wlim
            index_sum = index.sum(axis=2).sum(axis=0)
            for ki, s in enumerate(index_sum):
                s_lim = int(np.floor((1 - self.Wlim) / self.Wlim))
                if s > s_lim:
                    w_arg = self.W[:, ki, :].argsort()[::-1]
                    ind = cp.zeros_like(index[:, ki, :])
                    for ai in w_arg[:s_lim]:
                        ind[ai // n, ai % n] = True
                    self.W[:, ki, :] /= (~ind * self.W[:, ki, :]).sum() / (1 - s_lim * self.Wlim)
                    self.W[:, ki, :][ind] = self.Wlim
                else:
                    self.W[:, ki, :] /= (~index[:, ki, :] * self.W[:, ki, :]).sum() / (1 - s * self.Wlim)
                    self.W[:, ki, :][index[:, ki, :]] = self.Wlim
        lim = self.Vmax / self.Wlim * self.Hlim
        self.H[self.H > lim] = lim

    def _scale_H(self):
        lim = self.Vmax / self.Wlim * self.Hlim
        self.H[self.H > lim] = lim
    
    def _calc_u1_vu2(self):
        u1 = 1.0 / self.U
        vu2 = self.V * u1 * u1
        return u1, vu2

    def _update_U(self):
        
        self.U = conv(self.W, self.H, self.z_th)
        
    def _update_W(self, u1, vu2):
        _, t = self.V.shape
        l = self.l
        win = l - 1
        tb = t + win

        for li in range(l):
            self.W[:, :, li] *= cp.sqrt(cp.dot(vu2, self.H[:, win-li:tb-li].T) / cp.dot(u1, self.H[:, win-li:tb-li].T))
        
    def _update_H(self, u1, vu2):
            
        self.H *= cp.sqrt(iconv(self.W, vu2) / iconv(self.W, u1))

        
    def _recon_cost(self, calcU):
        
        if calcU:
            self._update_U()
        
        vu = self.V / self.U
        return cp.asnumpy(vu - np.log(vu) - 1).sum()
    
    def reconstruction_cost(self):
        
        return self._recon_cost(True)
        
    def solve(self, update_W=True):
        
        self.cost.append(self.reconstruction_cost())
        
        for i in range(self.max_iter):
            if update_W:
                Wold = self.W.copy()
            Hold = self.H.copy()
            
            self._update_U()
            u1, vu2 = self._calc_u1_vu2()
            if update_W:
                self._update_W(u1, vu2)
            self._update_H(u1, vu2)
            rc = self.reconstruction_cost()
            
            for j in range(10):
                    
                if rc > self.cost[-1] * (1 + 0.1 * (self.max_iter - i) / self.max_iter):
                    if update_W:
                        self.W = 0.5 * (self.W + Wold)
                    self.H = 0.5 * (self.H + Hold)
                    rc = self.reconstruction_cost()
                else:
                    break
                        
            if update_W:
                self._scale_WH()
            else:
                self._scale_H()
            rc = self.reconstruction_cost()
            self.cost.append(rc)
            if i >= 5:
                if abs(np.mean(cp.asnumpy(self.cost[-6:-1])) - rc) < self.tolerance:
                    break