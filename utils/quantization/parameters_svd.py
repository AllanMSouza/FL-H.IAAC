import  sys
import numpy as np
from sklearn.utils.extmath import randomized_svd

def parameter_svd_write(arrays, n_components):

    try:

        u = []
        vt = []
        sigma_parameters = []
        arrays_compre = []
        for i in range(len(arrays)):
            # print("Indice da camada: ", i)
            arrays_compre += parameter_svd(arrays[i], n_components)

        return arrays_compre

    except Exception as e:
        print("paramete_svd")
        print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, 0), type(e).__name__, e)

def parameter_svd(layer, n_components):

    try:
        if np.ndim(layer) == 1:
            return [layer, np.array([]), np.array([])]
        elif np.ndim(layer) == 2:
            r = svd(layer, n_components)
            return r
        elif np.ndim(layer) >= 3:
            u = []
            v = []
            sig = []
            for i in range(len(layer)):
                r = parameter_svd(layer[i], n_components)
                u.append(r[0])
                v.append(r[1])
                sig.append(r[2])
            return [np.array(u), np.array(v), np.array(sig)]

    except Exception as e:
        print("parameter_svd")
        print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, 0), type(e).__name__, e)


def svd(layer, n_components):

    try:
        np.random.seed(0)
        # print("ola: ", int(len(layer) * n_components), layer.shape, layer)
        U, Sigma, VT = randomized_svd(layer,
                                      n_components=int(len(layer) * n_components),
                                      n_iter=5,
                                      random_state=0)
        # U, Sigma, VT = np.linalg.svd(layer, full_matrices=True)
        # u = u[:, :n_components]
        # sigma = sigma[:n_components]
        # v = v[:threshold, :]

        # print(U.shape, Sigma.shape, VT.T.shape)
        return [U, VT, Sigma]

    except Exception as e:
        print("svd")
        print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, 0), type(e).__name__, e)


def inverse_parameter_svd_reading(arrays, model_shape):
    try:

        sketched_paramters = []
        reconstructed_model = []
        parameter_index = 0
        sig_ind = 0
        j = 0
        for i in range(len(model_shape)):
            layer_shape = model_shape[i]
            # print("i32: ", i*3+2)
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
                for j in range(len(u[i])):
                    layers_j.append(np.matmul(u[i][j] * sigma[i][j], v[i][j]))
                layers_l.append(layers_j)
            return np.array(layers_l)

    except Exception as e:
        print("inverse_parameter_svd")
        print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, 0), type(e).__name__, e)

def test():
    original = [np.random.random((4, 4, 4)) for i in range(1)]

    U, Sigma, VT = np.linalg.svd(original[0], full_matrices=True)
    U = U[:, :, :2]
    Sigma = np.expand_dims(Sigma[ :, 2], 1)
    VT = VT[:, :2, :].T

    print("compactado shapes: ", [i.shape for i in [U, Sigma, VT]])

    svd = parameter_svd_write(original, 0.5)
    original_ = inverse_parameter_svd_reading(svd, [i.shape for i in original])
    o = [U[0], Sigma.T, VT[0]]
    print("numpy shape: ", [i.shape for i in o])

    print("Original: \n", original, [i.shape for i in original])
    print("Np recontruido: \n", np.matmul(U*Sigma.T, VT.T))
    print("Reconstruído: \n", original_)


if __name__ == "__main__":
    test()