import numpy as np

def gen_data(k, l, t=3000, n_per_k=100, n_null=100, delta=0.25, z_th=0.001):
    
    n = k * n_per_k
    r = np.random.poisson(lam=0.05, size=(t, k))
    now = np.zeros(k)
    core = np.zeros((t, k))

    for i in range(t):
        now += r[i]
        now *= np.exp(-delta)
        now[now < z_th] = 0
        core[i] = now

    x = np.zeros((t, n))

    for ki in range(k):
        for i in range(n_per_k):
            ri = np.random.randint(l)
            if ri == 0:
                x[:, ki*n_per_k+i] = core[:, ki]
            else:
                x[ri:, ki*n_per_k+i] = core[:-ri, ki]
    
    y = x * np.random.exponential(scale=1, size=(t, n))
    null = np.random.choice(y.reshape(-1), size=(t, n_null))

    return np.hstack((y, null)) + y[y > 0].min()