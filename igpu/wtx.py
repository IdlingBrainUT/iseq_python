import numpy as np
import cupy as cp
from scipy.stats import norm

def calc_wtx_corr(wtx1, wtx2, l):
    corr_max = -cp.inf
    shift_max = 0
    for i in range(l*2-1):
        shift = i-(l-1)
        corr = cp.corrcoef(wtx1[l:-l], wtx2[l+shift:-l+shift])[0, 1]
        if corr > corr_max:
            corr_max = corr
            shift_max = shift
    return corr_max, shift_max

def calc_wtx_kendall(diff_y, diff_x, l):
    corr_max = 0
    shift_max = 0
    n = diff_y.shape[0] - 2 * l
    nn1 = n * (n - 1)
    v0 = nn1 * (2 * n + 5)
    diff_y_cut = diff_y[l:-l, l:-l]
    zt = int(cp.sum(diff_y_cut == 0)) - n
    t = (1+np.sqrt(1+4*zt))/2
    vt = (2 * t + 5) * zt
    for i in range(l*2-1):
        shift = i-(l-1)
        diff_x_cut = diff_x[l+shift:-l+shift, l+shift:-l+shift]
        diff_shift = diff_y_cut * diff_x_cut
        kp = int(cp.sum(diff_shift == 1))
        km = int(cp.sum(diff_shift == -1))
        zu = int(np.sum(diff_x_cut == 0)) - n
        u = (1+np.sqrt(1+4*zu))/2
        vu = (2 * u + 5) * zu
        v1 = zt / (2 * nn1) * zu
        v2 = zt / 9 * (t - 2) / (n - 2) * zu / nn1 * (u - 2)
        v = (v0 - vt - vu) / 18 + v1 + v2
        corr = norm.cdf((kp - km) / np.sqrt(v)) ** (l*2 - 1)
        if corr > corr_max:
            corr_max = corr
            shift_max = shift
    return corr_max, shift_max

def concat_w(model, i, j, shift, balance):
    bal1 = int(balance[i])
    bal2 = int(balance[j])
    bal_res1 = abs(balance[i] - bal1)
    bal_res2 = abs(balance[j] - bal2)
    if shift > 0:
        wi_shift = max(bal1, 0)
        wj_shift = max(-bal2, 0)
    else:
        wi_shift = max(-bal1, 0)
        wj_shift = max(bal2, 0)

    if shift >= wi_shift + wj_shift:
        res = int(shift - (wi_shift + wj_shift))
        res_half = res // 2
        wi_shift += res_half
        wj_shift += res_half
        if res % 2 == 1:
            if bal_res1 > bal_res2:
                wi_shift += 1
            else:
                wj_shift += 1
    else:
        res = int((wi_shift + wj_shift) - shift)
        res_half = res // 2
        wi_shift -= res_half
        wj_shift -= res_half
        if res % 2 == 1:
            if bal_res1 > bal_res2:
                wi_shift -= 1
            else:
                wj_shift -= 1
    if wi_shift < 0:
        wj_shift += -wi_shift
        wi_shift = 0
    if wj_shift < 0:
        wi_shift += -wj_shift
        wj_shift = 0

    w_new = cp.zeros_like(model.W[:, 0, :])
    data_num = cp.zeros(model.l)
    if shift > 0:
        if wi_shift == 0:
            w_new[:, :] += model.W[:, i, :]
            data_num += 1
        else:
            w_new[:, :-wi_shift] += model.W[:, i, wi_shift:]
            data_num[:-wi_shift] += 1
        if wj_shift == 0:
            w_new[:, :] += model.W[:, j, :]
            data_num[:] += 1
        else:
            w_new[:, wj_shift:] += model.W[:, j, :-wj_shift]
            data_num[wj_shift:] += 1
    else:
        if wi_shift == 0:
            w_new[:, :] += model.W[:, i, :]
            data_num += 1
        else:
            w_new[:, wi_shift:] += model.W[:, i, :-wi_shift]
            data_num[wi_shift:] += 1
        if wj_shift == 0:
            w_new[:, :] += model.W[:, i, :]
            data_num += 1
        else:
            w_new[:, :-wj_shift] += model.W[:, j, wj_shift:]
            data_num[:-wj_shift] += 1

    empty = data_num == 0
    w_new[:, empty] = model.z_th
    data_num[empty] = 1
    w_new /= data_num
    
    w_new /= w_new.sum()
    
    return w_new