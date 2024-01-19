import sys
import numpy as np
from scipy.stats import beta
import scipy

def pdf_beta_distribution(x, a, b, seed):

    try:

        # np.random.seed(seed)
        if (x < 1 and x > 0 and a > 0 and b > 0):

            vals = beta.pdf(x, a, b)

            return vals

        else:
            print("fora : ", x, a, b)
            return 0

    except Exception as e:
        print("pdf betadistribution")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

def estimate_params(Q):

    try:

        mean = np.mean(Q)
        var = np.var(Q)

        alpha = mean * (mean + var * var)
        beta = (1 - mean) * (mean + var * var)

        return alpha, beta

    except Exception as e:
        print("estimate params")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

def cda_fedavg_drift_detection(Q, lamda, delta, n_max, seed):
    """
    Compute the
    FedAvg drift
    :param Q: confidence list
    :param lamda:
    :param delta:
    :param n_max:
    """

    try:
        s_f = 0
        t_h = -np.log(lamda)
        N = len(Q)

        for k in range(delta, N - delta):

            m_b = np.mean(Q)
            m_a = np.mean(Q[k + 1:])
            # print("==========================")
            # print("m a: ", m_a, " m b: ", m_b, (1 - lamda) * m_b)
            #
            # print("m a: ", m_a, " m b: ", (1 - lamda) * m_b)

            if m_a <= (1 - lamda) * m_b:
                print("dent:")
                s_k = 0
                alpha_b, beta_b = estimate_params(Q[:k + 1])
                alpha_a, beta_a = estimate_params(Q[k + 1:])

                for i in range(k+1, N):

                    q = Q[i]
                    p1 = pdf_beta_distribution(q, alpha_a, beta_a, seed)
                    p2 = pdf_beta_distribution(q, alpha_b, beta_b, seed)
                    # print("p1: ", p1, " p2: ", p2, q, alpha_b, beta_b, seed)
                    # print("x: ", p1/p2)
                    if p2 == 0:
                        continue
                    else:
                        s_k = s_k + np.log(p1/p2)
                # print(" s k: ", s_k)
                s_f = np.max([s_f, s_k])

        print("s f: ", s_f)
        print("t h: ", t_h)
        if s_f > t_h:
            return True
        else:
            return False

    except Exception as e:
        print("cda fedavg drift detection")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)