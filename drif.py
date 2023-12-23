import random

import numpy as np
from river import drift
from scipy.stats import beta

def pdf_beta_distribution(x, a, b):

    if (x < 1 and x > 0 and a > 0 and b > 0):

        vals = beta.pdf(x, a, b)

        return vals

    else:

        return 0

def estimate_params(Q):

    mean = np.mean(Q)
    var = np.var(Q)

    alpha = mean * (mean + var * var)
    beta = (1 - mean) * (mean + var * var)

    return alpha, beta

def cda_fedavg_drift_detection(Q, lamda, delta, n_max):
    """
    Compute the
    FedAvg drift
    :param Q: confidence list
    :param lamda:
    :param delta:
    :param n_max:
    """
    s_f = 0
    t_h = -np.log(lamda)
    N = len(Q)

    for k in range(delta, N - delta):

        m_b = np.mean(Q)
        m_a = np.mean(Q[k + 1:])
        print("==========================")
        print("m a: ", m_a, " m b: ", m_b)

        print("m a: ", m_a, " m b: ", (1 - lamda) * m_b)

        if m_a <= (1 - lamda) * m_b:

            s_k = 0
            alpha_b, beta_b = estimate_params(Q[:k + 1])
            alpha_a, beta_a = estimate_params(Q[k + 1:])

            for i in range(k+1, N):

                q = Q[i]
                p1 = pdf_beta_distribution(q, alpha_a, beta_a)
                p2 = pdf_beta_distribution(q, alpha_b, beta_b)
                print("p1: ", p1, " p2: ", p2)
                print("x: ", p1/p2)
                s_k = s_k + np.log(p2/p1)
            print(" s k: ", s_k)
            s_f = np.max([s_f, s_k])

    print("s f: ", s_f)
    print("t h: ", t_h)
    if s_f > t_h:
        return True
    else:
        return False

rng = random.Random(12345)
adwin = drift.ADWIN()

# data_stream = rng.choices([0, 0.9], k=1000) + rng.choices(range(4, 8), k=1000)
data_stream = np.random.uniform(0, 1, size=100)
data_stream = np.concatenate((data_stream, np.random.uniform(0, 0.9, size=100)))
# print(data_stream)


data_stream = np.array(data_stream)

lamda = 0.05
delta = 50
n_max = len(data_stream)

detect = cda_fedavg_drift_detection(data_stream, lamda, delta, n_max)

print(detect)


# print(data_stream)