import copy

import torch
import numpy as np
from scipy.sparse import coo_array, csr_matrix, csc_matrix, bsr_matrix, csc_array, dia_array, dok_array, lil_array, coo_matrix, csr_array, bsr_array, dia_matrix, dok_matrix, lil_matrix
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

        elif type(sparse) == csr_array:
            return sparse.data.nbytes + sparse.indptr.nbytes + sparse.indices.nbytes

        elif type(sparse) == csr_matrix:
            return sparse.data.nbytes + sparse.indptr.nbytes + sparse.indices.nbytes

        elif type(sparse) == coo_array:
            return sparse.data.nbytes + sparse.row.nbytes + sparse.col.nbytes

        elif type(sparse) == coo_matrix:
            return sparse.data.nbytes + sparse.row.nbytes + sparse.col.nbytes

        elif type(sparse) == bsr_array:
            return sparse.data.nbytes  + sparse.indptr.nbytes + sparse.indices.nbytes

        elif type(sparse) == bsr_matrix:
            return sparse.data.nbytes  + sparse.indptr.nbytes + sparse.indices.nbytes

        elif type(sparse) == csc_array:
            return sparse.data.nbytes + sparse.indptr.nbytes + sparse.indices.nbytes

        elif type(sparse) == csc_matrix:
            return sparse.data.nbytes + sparse.indptr.nbytes + sparse.indices.nbytes

        elif type(sparse) == dia_matrix:
            return sparse.data.nbytes

        elif type(sparse) == dia_array:
            return sparse.data.nbytes

        elif type(sparse) == dok_array:
            return sparse.nbytes

        elif type(sparse) == dok_matrix:
            return sparse.data.nbytes + sparse.row.nbytes

        elif type(sparse) == lil_array:
            return sparse.data.nbytes + sparse.rows.nbytes

        elif type(sparse) == lil_matrix:
            return sparse.data.nbytes + sparse.rows.nbytes

        elif type(sparse) == np.ndarray:
            return sparse.nbytes

        else:
            print("nenhum: ", type(sparse))
            exit()

        # return sparse.data.nbytes + sparse.indptr.nbytes + sparse.indices.nbytes

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
            return lil_array(layer)
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

def sparse_bytes(sparse):

    try:

        bytes = 0
        if type(sparse) == list:

            for i in range(len(sparse)):
                bytes += sparse_bytes(sparse[i])

            return bytes

        elif type(sparse) == csr_matrix:
            return sparse.data.nbytes + sparse.indptr.nbytes + sparse.indices.nbytes

        elif type(sparse) == coo_array:
            return sparse.data.nbytes + sparse.row.nbytes + sparse.col.nbytes

        elif type(sparse) == coo_matrix:
            return sparse.data.nbytes + sparse.row.nbytes + sparse.col.nbytes

        elif type(sparse) == bsr_array:
            return sparse.data.nbytes + sparse.row.nbytes + sparse.col.nbytes

        elif type(sparse) == bsr_matrix:
            return sparse.data.nbytes + sparse.row.nbytes + sparse.col.nbytes

        elif type(sparse) == csc_array:
            return sparse.data.nbytes + sparse.row.nbytes + sparse.col.nbytes

        elif type(sparse) == csc_matrix:
            return sparse.data.nbytes + sparse.row.nbytes + sparse.col.nbytes

        elif type(sparse) == dia_matrix:
            return sparse.data.nbytes + sparse.row.nbytes + sparse.col.nbytes

        elif type(sparse) == dia_array:
            return sparse.data.nbytes + sparse.row.nbytes + sparse.col.nbytes

        elif type(sparse) == dok_array:
            return sparse.data.nbytes + sparse.row.nbytes + sparse.col.nbytes

        elif type(sparse) == dok_matrix:
            return sparse.data.nbytes + sparse.row.nbytes + sparse.col.nbytes

        elif type(sparse) == lil_array:
            return sparse.data.nbytes + sparse.rows.nbytes

        elif type(sparse) == lil_matrix:
            return sparse.data.nbytes + sparse.row.nbytes + sparse.col.nbytes

        elif type(sparse) == np.ndarray:
            return sparse.nbytes

        else:
            print("nenhum: ", type(sparse))

    except Exception as e:
        print("sparse bytes")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


def calculate_bytes(parameters):

    try:

        size = 0

        for p in parameters:

            sparse = sparse_matrix(p)
            # print("Tamanho original: ", p.nbytes)
            b = sparse_bytes(sparse)
            # print("Apos esparcificacao: ", b)
            b = min(p.nbytes, b)
            size += b
        return size

    except Exception as e:
        print("calculate bytes")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

def client_specific_top_k_parameters(client_id, parameters, clients_model_non_zero_indexes={}):
    if client_id in clients_model_non_zero_indexes:
        indexes_list = clients_model_non_zero_indexes[client_id]

        for i in range(len(parameters)):

            parameter = parameters[i]
            indexes = indexes_list[i]

            zeros = np.zeros(parameter.shape, dtype=np.double)

            if zeros.ndim == 1:
                # for j in range(len(indexes[0])):
                # 	zeros[indexes[0][j]] = parameter[indexes[0][j]]
                zeros = parameter

            elif zeros.ndim == 2:
                for j in range(len(indexes)):
                    for k in range(len(indexes[j])):
                        # print("valor: ", parameter[indexes[0][j], indexes[1][j]])
                        if indexes[j, k]:
                            parameter[j, k] = 0


            elif zeros.ndim == 3:
                for j in range(len(indexes)):
                    for k in range(len(indexes[j])):
                        for l in range(len(indexes[j, k])):
                            if indexes[j, k, l]:
                                parameter[j, k, l] = 0

            elif zeros.ndim == 4:
                for j in range(len(indexes)):
                    for k in range(len(indexes[j])):
                        for l in range(len(indexes[j, k])):
                            for m in range(len(indexes[j, k, l])):
                                if indexes[j, k, l, m]:
                                    parameter[j, k, l, m] = 0

            parameters[i] = parameter

    return parameters

def get_not_zero_values(updated_parameters, parameters, k_values):

    try:
        for i in range(len(updated_parameters)):

            updated_layer = updated_parameters[i]
            layer = copy.deepcopy(parameters[i])
            k_value = k_values[i]
            if k_value == -1:
                continue
            layer[updated_layer < k_value] = 0
            parameters[i] = layer


            # non_zero_indexes = np.argwhere(updated_layer >= k_value)
            # zero = np.zeros(updated_layer.shape)
            # size = len(non_zero_indexes)
            # for j in range(len(non_zero_indexes[0])):
            #         if size == 1:
            #             zero[non_zero_indexes[0][j]] = layer[non_zero_indexes[0][j]]
            #         elif size == 2:
            #             zero[non_zero_indexes[0][j], non_zero_indexes[1][j]] = layer[non_zero_indexes[0][j], non_zero_indexes[1][j]]
            #         elif size == 3:
            #             zero[non_zero_indexes[0][j], non_zero_indexes[1][j], non_zero_indexes[2][j]] = layer[non_zero_indexes[0][j], non_zero_indexes[1][j], non_zero_indexes[2][j]]
            #         elif size == 4:
            #             zero[non_zero_indexes[0][j], non_zero_indexes[1][j], non_zero_indexes[2][j], non_zero_indexes[3][j]] = layer[non_zero_indexes[0][j], non_zero_indexes[1][j], non_zero_indexes[2][j], non_zero_indexes[3][j]]
            #
            # parameters[i] = copy.copy(zero)

            # print("zero:")
            # print(zero)
            # exit()
            return parameters

    except Exception as e:
        print("get non zero values")
        print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


def client_model_non_zero_indexes(client_id, parameters, clients_model_non_zero_indexes, k=0.5):
    non_zero_indexes = []

    for p in parameters:
        zero = p == 0
        count = len(zero.flatten()==True)
        if count == len(p.flatten()):
            t, k_values = sparse_crs_top_k(np.abs(p), k)
            t = get_not_zero_values(np.abs(p), p, k_values)
            zero = t == 0
        non_zero_indexes.append(zero)

    clients_model_non_zero_indexes[client_id] = non_zero_indexes

    return clients_model_non_zero_indexes

# x = np.array([[1.3,2,3,4,5], [6,7,8,9,10], [6,7,8,9,10]])
# k = 0.6
# x2 = sparse_top_k(x, k)[0]
# print("numpy: ", x2.nbytes)
# x3 = csr_matrix(x2)
# print("csr matrix: ", bytes(x3))
# x4 = csr_array(x2)
# print("csr array: ", bytes(x4))
# x5 = bsr_matrix(x2)
# print("bsr matrix: ", bytes(x5))
# x6 = bsr_array(x2)
# print("bsr array: ", bytes(x6))
# x7 = bsr_matrix(x2)
# print("bsr matrix: ", bytes(x7))
# x8 = bsr_array(x2)
# print("bsr array: ", bytes(x8))
# x9 = dia_matrix(x2)
# print("dia matrix: ", bytes(x9))
# x10 = dia_array(x2)
# print("dia array: ", bytes(x10))
# x11 = dia_matrix(x2)
# print("dok matrix: ", bytes(x11))
# x12 = dok_array(x2)
# print("dok array: ", bytes(x12))
# x13 = lil_matrix(x2)
# print("lil matrix: ", bytes(x13))
# x14 = lil_array(x2)
# print("lil array: ", bytes(x14))
# x15 = csc_array(x2)
# print("csc array: ", bytes(x15))
# x16 = csc_matrix(x2)
# print("csc matrix: ", bytes(x16))
# x17 = coo_matrix(x2)
# print("coo matrix: ", bytes(x17))
# x18 = coo_array(x2)
# print("coo array: ", bytes(x18))
# print(quantize(x, 3))