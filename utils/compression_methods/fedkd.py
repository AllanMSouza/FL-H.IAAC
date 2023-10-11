import sys
import copy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from utils.compression_methods.parameters_svd import parameter_svd_write, inverse_parameter_svd_reading, if_reduces_size
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





def fedkd_compression_core(parameters, energy):
    try:
        for i in range(0, len(parameters), 3):
            u = parameters[i]
            v = parameters[i + 1]
            s = parameters[i + 2]

            if len(v) == 0:
                continue

            if len(u.shape) == 4:
                print("u shape: ", u.shape, "s shape: ", s.shape, " v shape: ", v.shape)
                u = np.transpose(u, (2, 3, 0, 1))
                s = np.transpose(s, (2, 0, 1))
                v = np.transpose(v, (2, 3, 0, 1))
            threshold = 1
            if np.sum(np.square(s)) == 0:
                continue
            else:
                for singular_value_num in range(len(s)):
                    print("lef: ", np.sum(np.square(s[:singular_value_num])) , " rig: ", energy * np.sum(np.square(s)))
                    if np.sum(np.square(s[:singular_value_num])) > energy * np.sum(np.square(s)):
                        print("aumentou: ", singular_value_num)
                        threshold = singular_value_num
                        break
                u = u[:, :threshold]
                s = s[:threshold]
                v = v[:threshold, :]
                # support high-dimensional CNN param
                if len(u.shape) == 4:
                    u = np.transpose(u, (2, 3, 0, 1))
                    s = np.transpose(s, (1, 2, 0))
                    v = np.transpose(v, (2, 3, 0, 1))

                parameters[i] = u
                parameters[i + 1] = v
                parameters[i + 2] = s

        for i in parameters:
            print(i.shape)

        return parameters

    except Exception as e:
        print("fedkd compression_methods core")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

def fedkd_compression(round_of_last_fit, layers_comppression_range, num_rounds, client_id, server_round, M, parameter):

    nt = server_round - round_of_last_fit
    layers_fraction = []
    n_components_list = []
    for i in range(M):
        # if i % 2 == 0:
        layer = parameter[i]
        if len(layer.shape) >= 2:

            if layers_comppression_range[i] > 0:
                n_components = layers_comppression_range[i]
            else:
                n_components = None
        else:
            n_components = None

        n_components_list.append(n_components)

    print("Vetor de componentes: ", n_components_list)

    parameter = parameter_svd_write(parameter, n_components_list)
    tmin = 0.01
    tmax = 0.95
    energy = tmin + (tmax-tmin)*(server_round/num_rounds)
    print("energy: ", energy)
    parameter = fedkd_compression_core(parameter, energy)

    # else:
    #     new_parameter = []
    #     for param in parameter:
    #         new_parameter.append(np.array(param))
    #         new_parameter.append(np.array([]))
    #         new_parameter.append(np.array([]))
    #
    #     parameter = copy.deepcopy(new_parameter)
    #
    #     layers_fraction = [1] * len(parameter)
    parameter = [np.array(i) for i in parameter]

    return parameter, layers_fraction