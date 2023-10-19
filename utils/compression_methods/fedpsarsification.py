import sys
import copy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from utils.compression_methods.parameters_svd import parameter_svd_write, inverse_parameter_svd_reading, if_reduces_size
from utils.compression_methods.fedkd import fedkd_compression
import os
from torch.nn.parameter import Parameter
import scipy.stats as st
# from torch_cka import CKA
import pandas as pd

import math
import torch
import numpy as np
from analysis.base_plots import line_plot, box_plot, ecdf_plot

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from scipy.sparse import coo_array, csr_matrix

def sparse_top_k(x, k):
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
    return csr_matrix(out_x.numpy())

def fedsparsification_server(parameters, k):

    for i in range(len(parameters)):

        parameters[i] = sparse_top_k(torch.from_numpy(parameters[i]), k)

    return parameters