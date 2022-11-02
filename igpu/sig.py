import cupy as cp

def sig(w, sig_abs=5, sig_rel=0.01):
    w_max = cp.asarray(w).max(axis=2)
    cell_max = w_max[:-100]
    null_max = w_max[-100:]
    th_abs = null_max.mean(axis=0) + sig_abs * null_max.std(axis=0)
    th_rel = cell_max.max(axis=0) * sig_rel
    
    return cp.min(cp.vstack([(cell_max > th_abs).sum(axis=0), (cell_max > th_rel).sum(axis=0)]), axis=0)

def sig_bool(w, sig_abs=5, sig_rel=0.01):
    w_max = cp.asarray(w).max(axis=2)
    cell_max = w_max[:-100]
    null_max = w_max[-100:]
    th_abs = null_max.mean(axis=0) + sig_abs * null_max.std(axis=0)
    bool1 = cell_max > th_abs
    bool2 = (cell_max >= cell_max.max(axis=0) * sig_rel)

    return bool1 & bool2

def remove_nonsig(model, sig_cells, n_sig=20):
    th_seq = sig_cells.sum(axis=0) >= n_sig
    k_sig = int(th_seq.sum())
    if k_sig == 0:
        w_mean = model.W.mean(axis=1)
        h_mean = model.H.mean(axis=0)
        n, l = w_mean.shape
        t = h_mean.shape[0]
        model.W = cp.zeros((n, 1, l))
        model.W[:, 0, :] = w_mean
        model.H = cp.zeros((1, t))
        model.H[0, :] = h_mean
        model.k = 1
    else:
        model.W[:-100, th_seq, :] += (sig_cells[:, ~th_seq].sum(axis=1) * model.z_th)[:, cp.newaxis, cp.newaxis]
        model.W = model.W[:, th_seq, :]
        model.H = model.H[th_seq]
        model.k = k_sig
    model.Wmax = model.W.max(axis=2).max(axis=0)

    return