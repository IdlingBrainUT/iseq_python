import numpy as np
import cupy as cp
import copy
from .conv import *

class iSeqSingle:
    
    Wact = None
    Wpos = None
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
    
    def __init__(self, V, W, H, max_iter=100, z_th=0.001, tolerance=1e-7, Hlim=3, Wlim=0.1, pre_computed=False):
        
        self.V = V
        n, t = V.shape
        self.Vmax = self.V.max()
        self.Hlim = Hlim
        self.Wlim = Wlim
        self.max_iter = int(max_iter)
        self.tolerance = tolerance
        self.z_th = z_th
        
        self.cost = []

        if pre_computed:
            k, t_h = H.shape
            l = t_h - t + 1
            self.k, self.l = k, l
            self.Wact = cp.vstack([W[i][:, 0] for i in range(k)]).T
            self.Wpos = cp.vstack([W[i][:, 1] for i in range(k)]).T
            self.H = H
        else:
            _, k, l = W.shape
            self.k, self.l = k, l
            self.Wact = cp.zeros((n, k))
            self.Wpos = cp.zeros((n, k))
            self.H = cp.zeros((k, t+l-1))

            win = l - 1
            win2 = 2 * l - 1
            for wi in range(k):
                u = cp.asnumpy(conv(cp.asarray(W[:, wi, :][:, cp.newaxis, :]), cp.asarray(H[wi][cp.newaxis, :]), z_th))
                corr = np.zeros((n, 2*l-1))
                top = u.sum(axis=1).argmax()
                for ni in range(n):
                    corr[ni, l-1] = np.corrcoef(u[ni], u[top])[0, 1]
                    for i in range(1, l):
                        corr[ni, l-1-i] = np.corrcoef(u[ni, :-i], u[top, i:])[0, 1]
                        corr[ni, l-1+i] = np.corrcoef(u[ni, i:], u[top, :-i])[0, 1]
                corr_a = corr.argmax(axis=1)
                count = [(corr_a == i).sum() for i in range(win2)]
                now = np.sum(count[:win])
                pos = np.zeros(win)
                for i in range(win):
                    now += count[i+win]
                    pos[i] = now
                    now -= count[i]
                start = pos.argmax()
                corr = cp.asarray(corr)
                self.Wpos[:, wi] = corr[:, start:start+l].argmax(axis=1)
                self.Wact[:, wi] = cp.asarray(u.sum(axis=1))
                self.Wact[:, wi] /= self.Wact[:, wi].sum()
                p_pre = W[:, wi, :][top].argmax()
                p_post = int(self.Wpos[top, wi])
                if p_pre == p_post:
                    self.H[wi, :] = H[wi, :]
                elif p_pre > p_post:
                    p_tmp = p_pre - p_post
                    self.H[wi, p_tmp:] = H[wi, :-p_tmp]
                    self.H[wi, :p_tmp] = z_th
                else:
                    p_tmp = p_post - p_pre
                    self.H[wi, :-p_tmp] = H[wi, p_tmp:]
                    self.H[wi, -p_tmp:] = z_th
        
        self._scale_WH()

    def _scale_WH(self):
        w_sum = self.Wact.sum(axis=0)
        self.H *= w_sum[:, cp.newaxis]
        self.Wact /= w_sum
        while (self.Wact > self.Wlim).sum() > 0:
            index = self.Wact >= self.Wlim
            index_sum = index.sum(axis=0)
            for ki, s in enumerate(index_sum):
                s_lim = int(np.floor((1 - self.Wlim) / self.Wlim))
                if s > s_lim:
                    w_arg = self.Wact.argsort()[::-1]
                    ind = cp.zeros_like(index[:, ki])
                    for ai in w_arg[:s_lim]:
                        ind[ai] = True
                    self.Wact[:, ki] /= (~ind * self.Wact[:, ki]).sum() / (1 - s_lim * self.Wlim)
                    self.Wact[:, ki][ind] = self.Wlim
                else:
                    self.Wact[:, ki] /= (~index[:, ki] * self.Wact[:, ki]).sum() / (1 - s_lim * self.Wlim)
                    self.Wact[:, ki][index[:, ki]] = self.Wlim
        lim = self.Vmax / self.Wlim * self.Hlim
        self.H[self.H > lim] = lim

    def _calc_u1_vu2(self):
        u1 = 1.0 / self.U
        vu2 = self.V * u1 * u1
        return u1, vu2

    def _update_U(self):
        
        self.U = conv_single(self.Wact, self.Wpos, self.H, self.l, self.z_th)
        
    def _update_W(self, u1, vu2):
        n, t = self.V.shape
        k, l = self.k, self.l

        tmp = cp.zeros((n, k))
        for ki in range(k):
            tmp_k = cp.zeros((n, t))
            for ni in range(n):
                start = l-1-self.Wpos[ni, ki]
                tmp_k[ni, :] = self.H[ki, start:start+t]
            tmp[:, ki] = (tmp_k * vu2).sum(axis=1) / (tmp_k * u1).sum(axis=1)
        self.Wact *= cp.sqrt(tmp)
        
    def _update_H(self, u1, vu2):
        n, t = self.V.shape
        k, l = self.k, self.l
        win = l - 1
        t_h = t + win

        tmp = cp.zeros((k, t_h))
        for ki in range(k):
            tmp_uk = cp.zeros((n, t_h))
            tmp_vk = cp.zeros((n, t_h))
            for ni in range(n):
                tmp_uk[ni, win - self.Wpos[ni, ki]:win - self.Wpos[ni, ki] + t] = u1[ni, :]
                tmp_vk[ni, win - self.Wpos[ni, ki]:win - self.Wpos[ni, ki] + t] = vu2[ni, :]
            tmp[ki, :] = ((tmp_vk * self.Wact[:, ki][:, cp.newaxis]).sum(axis=0) + self.z_th) / ((tmp_uk * self.Wact[:, ki][:, cp.newaxis]).sum(axis=0) + self.z_th)
        self.H *= cp.sqrt(tmp)    
        
    def _recon_cost(self, calcU):
        
        if calcU:
            self._update_U()
        
        vu = self.V / self.U
        return cp.asnumpy(vu - np.log(vu) - 1).sum()
    
    def reconstruction_cost(self):
        
        return self._recon_cost(True)
        
    def solve(self):
        
        self.cost.append(self.reconstruction_cost())
        
        for i in range(self.max_iter):
            Wold = self.Wact.copy()
            Hold = self.H.copy()

            self._update_U()
            u1, vu2 = self._calc_u1_vu2()
            self._update_W(u1, vu2)
            self._update_H(u1, vu2)
            rc = self.reconstruction_cost()
            
            
            for j in range(10):
                if rc > self.cost[-1] * (1 + 0.1 * (self.max_iter - i) / self.max_iter):
                    self.Wact = 0.5 * (self.Wact + Wold)
                    self.H = 0.5 * (self.H + Hold)
                    rc = self.reconstruction_cost()
                else:
                    break

            self._scale_WH()
            rc = self.reconstruction_cost()
            self.cost.append(rc)
            if i >= 5:
                if abs(np.mean(cp.asnumpy(self.cost[-6:-1])) - rc) < self.tolerance:
                    break

def conv_single(w_act, w_pos, h, l, z_th):
    n, k = w_act.shape
    _, h_t = h.shape
    win = l - 1
    u = cp.zeros((n, h_t - win))
    for ki in range(k):
        tmp = w_act[:, ki:ki+1] * h[ki, :]
        for ni in range(n):
            if w_pos[ni, ki] == 0:
                u[ni, :] += tmp[ni, win:]
            else:
                u[ni, :] += tmp[ni, win-w_pos[ni, ki]:-w_pos[ni, ki]]
    u[u < z_th] = z_th

    return u