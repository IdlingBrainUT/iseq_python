import cupy as cp

from .conv import *
from .core import iSeq
from .sig import *
from .single import iSeqSingle
from .wtx import *

def solve(V, k, l, z_th=0.001, tolerance=1e-7, n_iter=[30, 30, 10, 30], comp_rate=0.3, Wlim=0.1, Hlim=1.0, corr_max=0.95, random_seed=None):
    
    _, t = V.shape
    V1 = cp.array([V[:, i*l:(i+1)*l].mean(axis=1) for i in range(int(cp.ceil(t/l)))]).T
    model = iSeq(cp.asarray(V1), k, 2, max_iter=n_iter[0], z_th=z_th, tolerance=tolerance, random_seed=random_seed, Wlim=Wlim, Hlim=Hlim)
    model.solve()
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
    
    model = iSeq(cp.asarray(V2), k, l, max_iter=n_iter[1], z_th=z_th, tolerance=tolerance, random_seed=random_seed+1, Wlim=Wlim, Hlim=Hlim)
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
    
    model._scale_WH()
    model.solve()
        
    wtx = iconv(model.W, model.V)

    wtx_corr = cp.zeros((model.k, model.k))
    wtx_shift = cp.zeros_like(wtx_corr)

    for yi in range(model.k):
        for xi in range(yi+1, model.k):
            corr, shift = calc_wtx_corr(wtx[yi], wtx[xi], l)
            wtx_corr[yi, xi] = corr
            wtx_corr[xi, yi] = corr
            wtx_shift[yi, xi] = shift
            wtx_shift[xi, yi] = -shift
    
    while wtx_corr.max() > corr_max:
        w_mass = cp.asnumpy((model.W.sum(axis=0) * cp.arange(model.l)).sum(axis=1))
        balance = 2 * w_mass - model.l
        arg = wtx_corr.argmax()
        i = int(arg // model.k)
        j = int(arg % model.k)
        if i > j:
            tmpi = i
            i = j
            j = tmpi
    
        w_new = concat_w(model, i, j, wtx_shift[i, j], balance)
        wtx_new = iconv(w_new.reshape(-1, 1, model.l), model.V)[0]
        wtx_corr_new = cp.zeros(model.k-1)
        wtx_shift_new = cp.zeros(model.k-1)

        count = 0
        for ii in range(model.k):
            if ii == j:
                continue
            elif ii != i:
                corr, shift = calc_wtx_corr(wtx_new, wtx[ii], l)
                wtx_corr_new[count] = corr
                wtx_shift_new[count] = shift
            count += 1
    
        model.W = cp.concatenate((model.W[:, :j, :], model.W[:, j+1:, :]), axis=1)
        model.W[:, i, :] = w_new
        model.H = cp.vstack((model.H[:j], model.H[j+1:]))
    
        wtx = cp.vstack((wtx[:j], wtx[j+1:]))
        wtx[i] = wtx_new
    
        tmp = cp.vstack((wtx_corr[:j], wtx_corr[j+1:]))
        wtx_corr = cp.hstack((tmp[:, :j], tmp[:, j+1:]))
        wtx_corr[i] = wtx_corr_new
        wtx_corr[:, i] = wtx_corr_new
    
        tmp = cp.vstack((wtx_shift[:j], wtx_shift[j+1:]))
        wtx_shift = cp.hstack((tmp[:, :j], tmp[:, j+1:]))
        wtx_shift[i] = wtx_shift_new
        wtx_shift[:, i] = wtx_shift_new
    
        model.k -= 1
    
    model.max_iter = n_iter[2]
    model.cost = []
    model._scale_WH()
    model.solve()
    
    h2 = model.H.copy()
    w2 = model.W.copy()
    
    model = iSeq(cp.asarray(V), model.k, l, max_iter=n_iter[3], z_th=z_th, tolerance=tolerance, random_seed=random_seed+2, Wlim=Wlim, Hlim=Hlim)
    model.W = w2
    
    count = 1
    for i, b in enumerate(t1_index):
        if i == 0:
            if b:
                model.H[:, :l-1] = h2[:, :l-1]
            else:
                model.H[:, :l-1] = model.z_th
        if b:
            model.H[:, l*(i+1)-1:l*(i+2)-1] = h2[:, l*count-1:l*(count+1)-1]
            count += 1
        else:
            model.H[:, l*(i+1)-1:l*(i+2)-1] = model.z_th
    
    model._scale_WH()
    model.solve()
    
    return model

def solve_single(V, k, l, z_th=0.001, tolerance=1e-7, n_iter=[30, 30, 10, 30], comp_rate=0.3, Wlim=0.1, Hlim=1.0, corr_max=0.95, random_seed=None):
    
    _, t = V.shape
    V1 = cp.array([V[:, i*l:(i+1)*l].mean(axis=1) for i in range(int(cp.ceil(t/l)))]).T
    model = iSeq(cp.asarray(V1), k, 2, max_iter=n_iter[0], z_th=z_th, tolerance=tolerance, random_seed=random_seed, Wlim=Wlim, Hlim=Hlim)
    model.solve()
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
    
    model = iSeq(cp.asarray(V2), k, l, max_iter=n_iter[1], z_th=z_th, tolerance=tolerance, random_seed=random_seed+1, Wlim=Wlim, Hlim=Hlim)
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
    
    model._scale_WH()
    model.solve()
        
    wtx = iconv(model.W, model.V)

    wtx_corr = cp.zeros((model.k, model.k))
    wtx_shift = cp.zeros_like(wtx_corr)
    for yi in range(model.k):
        for xi in range(yi+1, model.k):
            corr, shift = calc_wtx_corr(wtx[yi], wtx[xi], l)
            wtx_corr[yi, xi] = corr
            wtx_corr[xi, yi] = corr
            wtx_shift[yi, xi] = shift
            wtx_shift[xi, yi] = -shift
    
    while wtx_corr.max() > corr_max:
        w_mass = cp.asnumpy((model.W.sum(axis=0) * cp.arange(model.l)).sum(axis=1))
        balance = 2 * w_mass - model.l
        arg = wtx_corr.argmax()
        i = int(arg // model.k)
        j = int(arg % model.k)
        if i > j:
            tmpi = i
            i = j
            j = tmpi
    
        w_new = concat_w(model, i, j, wtx_shift[i, j], balance)
        wtx_new = iconv(w_new.reshape(-1, 1, model.l), model.V)[0]
        wtx_corr_new = cp.zeros(model.k-1)
        wtx_shift_new = cp.zeros(model.k-1)

        count = 0
        for ii in range(model.k):
            if ii == j:
                continue
            elif ii != i:
                corr, shift = calc_wtx_corr(wtx_new, wtx[ii], l)
                wtx_corr_new[count] = corr
                wtx_shift_new[count] = shift
            count += 1
    
        model.W = cp.concatenate((model.W[:, :j, :], model.W[:, j+1:, :]), axis=1)
        model.W[:, i, :] = w_new
        model.H = cp.vstack((model.H[:j], model.H[j+1:]))
    
        wtx = cp.vstack((wtx[:j], wtx[j+1:]))
        wtx[i] = wtx_new
    
        tmp = cp.vstack((wtx_corr[:j], wtx_corr[j+1:]))
        wtx_corr = cp.hstack((tmp[:, :j], tmp[:, j+1:]))
        wtx_corr[i] = wtx_corr_new
        wtx_corr[:, i] = wtx_corr_new
    
        tmp = cp.vstack((wtx_shift[:j], wtx_shift[j+1:]))
        wtx_shift = cp.hstack((tmp[:, :j], tmp[:, j+1:]))
        wtx_shift[i] = wtx_shift_new
        wtx_shift[:, i] = wtx_shift_new
    
        model.k -= 1
    
    model.max_iter = n_iter[2]
    model.cost = []
    model._scale_WH()
    model.solve()

    h2 = cp.zeros((model.k, t + model.l - 1))
    count = 1
    for i, b in enumerate(t1_index):
        if i == 0:
            if b:
                h2[:, :l-1] = model.H[:, :l-1]
            else:
                h2[:, :l-1] = model.z_th
        if b:
            h2[:, l*(i+1)-1:l*(i+2)-1] = model.H[:, l*count-1:l*(count+1)-1]
            count += 1
        else:
            h2[:, l*(i+1)-1:l*(i+2)-1] = model.z_th

    model = iSeqSingle(cp.asarray(V), model.W, h2, max_iter=n_iter[3], z_th=z_th, tolerance=tolerance, Wlim=Wlim, Hlim=Hlim)
    
    model._scale_WH()
    model.solve()
    
    return model

def wtx_demo(V, k, l, z_th=0.001, tolerance=1e-7, n_iter=[30, 30, 10, 30], comp_rate=0.3, Wlim=0.1, Hlim=1.0, corr_max=0.95, random_seed=None):
    
    _, t = V.shape
    V1 = cp.array([V[:, i*l:(i+1)*l].mean(axis=1) for i in range(int(cp.ceil(t/l)))]).T
    model = iSeq(cp.asarray(V1), k, 2, max_iter=n_iter[0], z_th=z_th, tolerance=tolerance, random_seed=random_seed, Wlim=Wlim, Hlim=Hlim)
    model.solve()
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
    
    model = iSeq(cp.asarray(V2), k, l, max_iter=n_iter[1], z_th=z_th, tolerance=tolerance, random_seed=random_seed+1, Wlim=Wlim, Hlim=Hlim)
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
    
    model._scale_WH()
    model.solve()
        
    wtx = iconv(model.W, model.V)
    wtx_copy = wtx.copy()

    wtx_corr = cp.zeros((model.k, model.k))
    wtx_shift = cp.zeros_like(wtx_corr)

    for yi in range(model.k):
        for xi in range(yi+1, model.k):
            corr, shift = calc_wtx_corr(wtx[yi], wtx[xi], l)
            wtx_corr[yi, xi] = corr
            wtx_corr[xi, yi] = corr
            wtx_shift[yi, xi] = shift
            wtx_shift[xi, yi] = -shift

    wtx_corr_copy = wtx_corr.copy()
    
    while wtx_corr.max() > corr_max:
        w_mass = cp.asnumpy((model.W.sum(axis=0) * cp.arange(model.l)).sum(axis=1))
        balance = 2 * w_mass - model.l
        arg = wtx_corr.argmax()
        i = int(arg // model.k)
        j = int(arg % model.k)
        if i > j:
            tmpi = i
            i = j
            j = tmpi
    
        w_new = concat_w(model, i, j, wtx_shift[i, j], balance)
        wtx_new = iconv(w_new.reshape(-1, 1, model.l), model.V)[0]
        wtx_corr_new = cp.zeros(model.k-1)
        wtx_shift_new = cp.zeros(model.k-1)

        count = 0
        for ii in range(model.k):
            if ii == j:
                continue
            elif ii != i:
                corr, shift = calc_wtx_corr(wtx_new, wtx[ii], l)
                wtx_corr_new[count] = corr
                wtx_shift_new[count] = shift
            count += 1
    
        model.W = cp.concatenate((model.W[:, :j, :], model.W[:, j+1:, :]), axis=1)
        model.W[:, i, :] = w_new
        model.H = cp.vstack((model.H[:j], model.H[j+1:]))
    
        wtx = cp.vstack((wtx[:j], wtx[j+1:]))
        wtx[i] = wtx_new
    
        tmp = cp.vstack((wtx_corr[:j], wtx_corr[j+1:]))
        wtx_corr = cp.hstack((tmp[:, :j], tmp[:, j+1:]))
        wtx_corr[i] = wtx_corr_new
        wtx_corr[:, i] = wtx_corr_new
    
        tmp = cp.vstack((wtx_shift[:j], wtx_shift[j+1:]))
        wtx_shift = cp.hstack((tmp[:, :j], tmp[:, j+1:]))
        wtx_shift[i] = wtx_shift_new
        wtx_shift[:, i] = wtx_shift_new
    
        model.k -= 1
       
    return (wtx, wtx_copy, wtx_corr, wtx_corr_copy)