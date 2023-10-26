import copy

import torch
import numpy as np
from scipy.sparse import coo_array, csr_matrix
import sys

'''
**********************************************
Input must be a pytorch tensor
**********************************************
'''

def bytes(sparse):

    try:
        if "numpy" in str(type(sparse)):
            return sparse.nbytes

        return sparse.data.nbytes + sparse.indptr.nbytes + sparse.indices.nbytes

    except Exception as e:
        print("bytes")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

def quantize(x, d):
    """quantize the tensor x in d level on the absolute value coef wise"""
    norm = np.sqrt(np.sum(np.square(x)))
    level_float = d * np.abs(x) / norm
    previous_level = np.floor(level_float)
    is_next_level = np.random.rand(*x.shape) < (level_float - previous_level)
    new_level = previous_level + is_next_level
    return np.sign(x) * norm * new_level / d

def quantize(x, n):
    # assume that x is a torch tensor
    # print('n:{}'.format(n))
    x = torch.from_numpy(x)
    x = x.float()
    x_norm = torch.norm(x, p=float('inf'))

    sgn_x = ((x > 0).float() - 0.5) * 2

    p = torch.div(torch.abs(x), x_norm)
    renormalize_p = torch.mul(p, n)
    floor_p = torch.floor(renormalize_p)
    compare = torch.rand_like(floor_p)
    final_p = renormalize_p - floor_p
    margin = (compare < final_p).float()
    xi = (floor_p + margin) / n

    Tilde_x = x_norm * sgn_x * xi

    return Tilde_x.numpy()


def sparse_randomized(x, input_compress_settings={}):
    max_iteration = 10000
    compress_settings = {'p': 0.8}
    compress_settings.update(input_compress_settings)
    # p=compress_settings['p']
    # vec_x=x.flatten()
    # out=torch.dropout(vec_x,1-p,train=True)
    # out=out/p
    vec_x = x.flatten()
    d = int(len(vec_x))
    p = compress_settings['p']

    abs_x = torch.abs(vec_x)
    # d=torch.prod(torch.Tensor(x.size()))
    out = torch.min(p * d * abs_x / torch.sum(abs_x), torch.ones_like(abs_x))
    i = 0
    while True:
        i += 1
        # print(i)
        if i >= max_iteration:
            raise ValueError('Too much operations!')
        temp = out.detach()

        cI = 1 - torch.eq(out, 1).float()
        c = (p * d - d + torch.sum(cI)) / torch.sum(out * cI)
        if c <= 1:
            break
        out = torch.min(c * out, torch.ones_like(out))
        if torch.sum(1 - torch.eq(out, temp)):
            break

    z = torch.rand_like(out)
    out = vec_x * (z < out).float() / out

    out = out.reshape(x.shape)

    # out=out.reshape(x.shape)
    return out


def one_bit(x, input_compress_settings={}):
    x_norm = torch.norm(x, p=float('inf'))
    sgn_x = ((x > 0).float() - 0.5) * 2

    compressed_x = x_norm * sgn_x

    return compressed_x


def top_k(x, k):

    try:
        # print("spars: ", type(x))
        if type(x) != type(np.array([])):
            print("tentou: ", type(x))
            x = np.array(x)
            # print("passou")
        entrada = copy.deepcopy(x)
        x = torch.from_numpy(x)
        vec_x = x.flatten()
        d = int(len(vec_x))
        # print(d)
        k = int(np.ceil(d * k))
        # print(k)
        indices = torch.abs(vec_x).topk(k)[1]
        out_x = torch.zeros_like(vec_x)
        out_x[indices] = vec_x[indices]
        out_x = out_x.reshape(x.shape)
        # print(x.shape)
        out = out_x.numpy()
        out_aux = np.abs(copy.copy(out))
        threshold = np.min(out_aux[out_aux > 0])
        return out, threshold

    except Exception as e:
        print("top k")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

def sparse_top_k(layer, k):

    try:
        if layer.ndim == 1:
            return layer, -1
        else:
            return top_k(layer, k)
        # else:
        #     new_layer = []
        #     k_values = []
        #     if layer.ndim == 3:
        #         for i in range(len(layer)):
        #             l, k_value = (sparse_top_k(layer[i], k))
        #             new_layer.append(l)
        #             k_values.append(k_value)
        #
        #     elif layer.ndim == 4:
        #         for i in range(len(layer)):
        #             row = []
        #             row_k_values = []
        #             for j in range(len(layer[i])):
        #                 p, k_value = sparse_top_k(layer[i][j], k)
        #                 row.append(p)
        #                 row_k_values.append(k_value)
        #             new_layer.append(row)
        #             k_values.append(row_k_values)
        #
        #     return np.array(new_layer), k_values

    except Exception as e:
        print("sparse top k")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

def sparse_crs_top_k(parameters, k):

    try:
        k_values = []
        for i in range(len(parameters)):
            p, k_value = sparse_top_k(parameters[i], k)
            # print(parameters[i].shape, " convertido para ", p.shape)
            # parameters[i] = sparse_matrix(p)
            parameters[i] = p
            k_values.append(k_value)
        return parameters, k_values
    except Exception as e:
        print("sparse crs top k")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

def sparse_matrix(layer):

    try:

        if layer.ndim == 1:
            return layer
        elif layer.ndim == 2:
            return csr_matrix(layer)
        else:
            new_layer = []
            if layer.ndim == 3:
                for i in range(len(layer)):
                    new_layer.append(sparse_matrix(layer[i]))
            elif layer.ndim == 4:
                for i in range(len(layer)):
                    row = []
                    for j in range(len(layer[i])):
                        row.append(sparse_matrix(layer[i][j]))
                    new_layer.append(row)

            return new_layer

    except Exception as e:
        print("sparse matrix")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

def to_dense(x):

    try:

        for i in range(len(x)):
            if type(x[i]) == csr_matrix:
                x[i] = x[i].toarray()

        return x

    except Exception as e:
        print("to dense")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

x = np.array([[1.3,2,3,4,5], [6,7,8,9,10], [6,7,8,9,10]])
k = 0.5
x2 = sparse_top_k(x, k)[0]
print(x2.nbytes)
x3 = csr_matrix(x2)
print(bytes(x3))
# print(quantize(x, 3))