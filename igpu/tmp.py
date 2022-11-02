import numpy as np
import pandas as pd

import cupy as cp

from .conv import *
from .core import iSeq
from .sig import *
from .wtx import *

def open_csv(filepath, n_null=100, random_seed=None):
    np.random.seed(random_seed)
    x = pd.read_csv(filepath, header=None).values[:, :-100]
    t, n_cell = x.shape
    x /= x.sum(axis=0)
    null = np.zeros((t, n_null))
    pool = x.reshape(-1)
    for i in range(n_null):
        null[:, i] = np.random.choice(pool, size=t)
        null[:, i] /= null[:, i].sum()
    y = np.hstack((x, null))
    y[y == 0] = y[y>0].min()

    return y

"""
def open_csv_nonull(filepath, n_null=100):
    x = pd.read_csv(filepath, header=None).values[:, :-n_null]
    x /= x.sum(axis=0)
    x[x == 0] = x[x > 0].min()
    return x
"""

def open_csv_nonull(filepath):
    x = pd.read_csv(filepath, header=None).values
    x[x == 0] = x[x > 0].min()
    return x

def solve_middle(V, k, l, z_th=0.001, tolerance=1e-7, n_iter=[30, 30, 10, 30], comp_rate=0.3, random_seed=None):
    
    n, t = V.shape
    V1 = cp.array([V[:, i*l:(i+1)*l].mean(axis=1) for i in range(int(cp.ceil(t/l)))]).T
    model = iSeq(cp.asarray(V1), k, 2, max_iter=n_iter[0], z_th=z_th, tolerance=tolerance, random_seed=random_seed)
    for i in range(n_iter[0]):
        model._update_W(True)
        model._update_H(True)
    h1 = model.H.copy()
    w1 = model.W.copy()
    
    t1 = V1.shape[1]
    t1_true = int(cp.ceil(t1 * comp_rate))
    t1_index = cp.array([False for _ in range(t1)])
    count = 0
    tmp = model.H.argsort(axis=1)[:, ::-1]

    for i in range(t1):
        for j in range(k):
            ij = tmp[j, i]
            if ij > 0 and ~t1_index[ij-1]:
                t1_index[ij-1] = True
                count += 1
            if ij > 1 and ~t1_index[ij-2]:
                t1_index[ij-2] = True
                count += 1
            if ij < t1 and ~t1_index[ij]:
                t1_index[ij] = True
                count += 1
        if count >= t1_true:
            break
    
    V2 = cp.asarray(V)[:, cp.array([t1_index for _ in range(l)]).T.reshape(-1)[:t]]
    
    model = iSeq(cp.asarray(V2), k, l, max_iter=n_iter[1], z_th=z_th, tolerance=tolerance, random_seed=random_seed+1)
    model.W = cp.zeros_like(model.W)
    model.W += w1.max(axis=2)[:, :, cp.newaxis]
    
    count = 0
    for i, b in enumerate(t1_index):
        if b:
            if count == 0:
                model.H[:, :l-1] = h1[:, i:i+1]
                count += 1
            model.H[:, l*count-1:l*(count+1)-1] = h1[:, i+1:i+2]
            count += 1
    
    w_sum = model.W.sum(axis=2).sum(axis=0)
    model.H *= w_sum.reshape(-1, 1) 
    model.W /= w_sum.reshape(1, -1, 1)  
    
    model.solve()
    
    return model