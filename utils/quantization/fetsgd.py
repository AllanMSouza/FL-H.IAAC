import numpy as np

from scipy import linalg

from scipy import sparse
import sys

def sparse_nbyes(sparse):

    return sparse.data.nbytes + sparse.indptr.nbytes + sparse.indices.nbytes

m = np.int8
rng = np.random.default_rng()
np.random.seed(0)

def sketching(array, sketch_n_rows):

    n_rows = len(array)
    sk = linalg.clarkson_woodruff_transform(array, sketch_n_rows)
    return sk, n_rows

def layers_sketching(arrays, sketch_n_rows=None, model_shape=None):

    try:

        sketched_paramters = []
        for i in range(len(arrays)):
            print("Indice da camada: ", i)
            if model_shape is not None:
                shape = model_shape[i]
            else:
                shape = None
            sketched_paramters.append(layer_sketc(arrays[i], sketch_n_rows, shape, 0))
            # CNN
            # if np.ndim(client_layer) == 4:
            #     n_rows = []
            #     sketched_cnn_layer = []
            #     for k in range(len(client_layer)):
            #         client_layer_k = client_layer[k]
            #         sketched_k_layer = []
            #         for j in range(len(client_layer_k)):
            #             client_layer_j = client_layer_k[j]
            #             if model_shape is not None:
            #                 sketch_n_rows = model_shape[i][k][j]
            #             client_layer_j_sketched, n_row = sketching(client_layer_j, sketch_n_rows)
            #             sketched_k_layer.append(client_layer_j_sketched)
            #
            #     sketched_layer.append(np.array(sketched_cnn_layer))

        return sketched_paramters

    except Exception as e:
        print("Set parameters to model")
        print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, 0), type(e).__name__, e)

def layer_sketc(layer, sketch_n_rows=None, model_shape=None, dim=0):

    try:
        if np.ndim(layer) == 1:
            return layer
        elif np.ndim(layer) == 2:
            if model_shape is not None:
                sketch_n_rows = model_shape[dim]
            print("teste: ", layer.shape, sketch_n_rows, dim)
            sk, n_row = sketching(layer, sketch_n_rows)
            return np.array(sk)
        elif np.ndim(layer) >= 3:
            layers_l = []
            for i in range(len(layer)):
                layers_l.append(layer_sketc(layer[i], sketch_n_rows, model_shape, dim+1))
            return np.array(layers_l)

    except Exception as e:
        print("Set parameters to model")
        print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, 0), type(e).__name__, e)

a = [np.random.random((31, 1, 32, 32))]
ak = layers_sketching(a, 10)
ak_r = layers_sketching(ak, 32)
print([i.shape for i in layers_sketching(a, 10)])
print([i.shape for i in layers_sketching(ak, 32)])