import torch
import numpy as np
from scipy.sparse import coo_array, csr_matrix

'''
**********************************************
Input must be a pytorch tensor
**********************************************
'''

def bytes(sparse):

    if "numpy" in str(type(sparse)):
        return sparse.nbytes

    return sparse.data.nbytes + sparse.indptr.nbytes + sparse.indices.nbytes

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


def sparse_top_k(x, k):
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
    return out_x.numpy()

def sparse_crs_top_k(parameters, k):

    for i in range(len(parameters)):
        p = sparse_top_k(parameters[i], k)
        parameters[i] = csr_matrix(p).toarray()
    return parameters

def to_dense(x):

    return x.toarray()

# x = np.array([[1,2,3,4,5], [6,7,8,9,10]])
# k = 1/100
# x2 = sparse_top_k(x, k)
# print(x2.nbytes)
# x3 = csr_matrix(x2)
# print(bytes(x3))
# print(quantize(x, 3))