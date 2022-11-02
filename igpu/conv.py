import cupy as cp

def conv(A, B, z_th):
    n, _, l = A.shape
    _, tb = B.shape
    ret = cp.zeros((n, tb-l+1))
    for li in range(l):
        ret += cp.dot(A[:, :, li], B[:, l-1-li:tb-li])
    ret += z_th
    return ret

def iconv(A, B):
    n, k, l = A.shape
    _, t = B.shape
    ret = cp.zeros((k, t+l-1))
    for li in range(l):
        ret[:, l-1-li:t+l-1-li*2] += cp.dot(A[:, :, li].T, B[:, :t-li])
    return ret