import  sys
import numpy as np
from sklearn.utils.extmath import randomized_svd

def parameter_svd_write(arrays, n_components_list, svd_type='tsvd'):

    try:

        u = []
        vt = []
        sigma_parameters = []
        arrays_compre = []
        for i in range(len(arrays)):
            if type(n_components_list) == list:
                n_components = n_components_list[i]
            else:
                n_components = n_components_list
            # print("Indice da camada: ", i)
            r = parameter_svd(arrays[i], n_components, svd_type)
            arrays_compre += r

        return arrays_compre

    except Exception as e:
        print("paramete_svd")
        print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, 0), type(e).__name__, e)

def parameter_svd(layer, n_components, svd_type='tsvd'):

    try:
        if np.ndim(layer) == 1 or n_components is None:
            return [layer, np.array([]), np.array([])]
        elif np.ndim(layer) == 2:
            r = svd(layer, n_components, svd_type)
            return r
        elif np.ndim(layer) >= 3:
            u = []
            v = []
            sig = []
            for i in range(len(layer)):
                r = parameter_svd(layer[i], n_components, svd_type)
                u.append(r[0])
                v.append(r[1])
                sig.append(r[2])
            return [np.array(u), np.array(v), np.array(sig)]

    except Exception as e:
        print("parameter_svd")
        print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, 0), type(e).__name__, e)


def svd(layer, n_components, svd_type='tsvd'):

    try:
        np.random.seed(0)
        # print("ola: ", int(len(layer) * n_components), layer.shape, layer)
        if n_components > 0 and n_components < 1:
            n_components = int(len(layer) * n_components)

        if svd_type == 'tsvd':
            U, Sigma, VT = randomized_svd(layer,
                                          n_components=n_components,
                                          n_iter=5,
                                          random_state=0)
        else:
            U, Sigma, VT = np.linalg.svd(layer, full_matrices=False)
            U = U[:, :n_components]
            Sigma = Sigma[:n_components]
            VT = VT[:n_components, :]

        # print(U.shape, Sigma.shape, VT.T.shape)
        return [U, VT, Sigma]

    except Exception as e:
        print("svd")
        print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, 0), type(e).__name__, e)

def if_reduces_size(shape, n_components, dtype=np.float64):

    try:
        size = np.array([1], dtype=dtype)
        p = shape[0]
        q = shape[1]
        k = n_components

        if p*k + k*k + k*q < p*q:
            return True
        else:
            return False

    except Exception as e:
        print("svd")
        print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, 0), type(e).__name__, e)


def inverse_parameter_svd_reading(arrays, model_shape, M=0):
    try:
        # print("recebidos aki: ", [i.shape for i in arrays])
        if M == 0:
            M = len(model_shape)
        sketched_paramters = []
        reconstructed_model = []
        parameter_index = 0
        sig_ind = 0
        j = 0
        for i in range(M):
            layer_shape = model_shape[i]
            # print("i32: ", i*3+2)
            # print("valor i: ", i, i*3, len(model_shape), len(arrays), "valor de M: ", M)
            u = arrays[i*3]
            v = arrays[i*3 + 1]

            si = arrays[i*3 + 2]
            # print("teste", u.shape, v.shape, si.shape, layer_shape)
            # print("maior: ", i*3 + 2, len(arrays))
            if len(layer_shape) == 1:
                parameter_layer = inverse_parameter_svd(u, v, layer_shape)
            else:
                parameter_layer = inverse_parameter_svd(u, v, layer_shape, si)
            if parameter_layer is None:
                # print("Pos ", i, i*3)
                pass
            reconstructed_model.append(parameter_layer)

        return reconstructed_model

    except Exception as e:
        print("inverse_paramete_svd")
        print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, 0), type(e).__name__, e)


def inverse_parameter_svd(u, v, layer_index, sigma=None, sig_ind=None):
    try:
        if len(v) == 0:
            return u
        if len(layer_index) == 1:
            # print("u1")
            return u
        elif len(layer_index) == 2:
            # print("u2")
            return np.matmul(u * sigma, v)
        elif len(layer_index) == 3:
            # print("u3")
            layers_l = []
            for i in range(len(u)):
                layers_l.append(np.matmul(u[i] * sigma[i], v[i]))
            return np.array(layers_l)
        elif len(layer_index) == 4:
            layers_l = []
            # print("u4")
            for i in range(len(u)):
                layers_j = []
                # print("u shape: ", u.shape, " v shape: ", v.shape)
                for j in range(len(u[i])):
                    layers_j.append(np.matmul(u[i][j] * sigma[i][j], v[i][j]))
                layers_l.append(layers_j)
            return np.array(layers_l)

    except Exception as e:
        print("inverse_parameter_svd")
        print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, 0), type(e).__name__, e)

def test():
    original = [np.random.random((5, 5)) for i in range(1)]

    threshold = 2
    U, Sigma, VT = np.linalg.svd(original[0], full_matrices=False)
    print("compactado shapes 1: ", [i.shape for i in [U, Sigma, VT]])
    U = U[:, :threshold]
    Sigma = Sigma[ : threshold]
    VT = VT[:threshold, :]

    print("compactado shapes 2: ", [i.shape for i in [U, Sigma, VT]])
    print(original[0])

    # svd = parameter_svd_write(original, 0.5)
    # original_ = inverse_parameter_svd_reading(svd, [i.shape for i in original])
    # o = [U[0], Sigma.T, VT[0]]
    # print("numpy shape: ", [i.shape for i in o])
    #
    # print("Original: \n", original, [i.shape for i in original])
    # print("Np recontruido: \n", np.matmul(U*Sigma.T, VT.T))
    # print("Reconstruído: \n", original_)
    print("Reconstruido")
    r = np.matmul(U * Sigma[..., None, :], VT)
    print(r)


if __name__ == "__main__":
    test()