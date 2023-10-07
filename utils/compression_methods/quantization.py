import numpy as np
# from bitstring import BitArray
import sys
import flwr


def quantize_linear_symmetric(value, bits):
    qmin = -2**(bits - 1)
    qmax = 2**(bits - 1) - 1
    scale = max(abs(qmin), abs(qmax)) / max(abs(np.min(value)), abs(np.max(value)))
    value = np.round(value * scale)
    value = np.clip(value, qmin, qmax).astype(int)
    print("Parameters after compression_methods: ", value)

    return value

def quantize_linear_asymmetric(value, bits):
    qmin = -2**(bits - 1)
    qmax = 2**(bits - 1) - 1
    scale = (qmax - qmin) / (np.max(value) - np.min(value))
    zero_point = qmin - np.min(value) * scale
    value = np.round(value * scale + zero_point)
    value = np.clip(value, qmin, qmax)
    return value

def min_max_quantization(w, b):

    try:

        print("or: ", w[0])
        if np.sum(w) == 0:
            return [w.astype(np.int8), 0, 0]
        mini = np.minimum(w.min(), 0)
        maxim = np.maximum(w.max(), 0)
        w = np.clip(w, mini, maxim)
        # print("de: ", w)
        s = (maxim - mini)/(2**b - 1)
        qmin = 0
        z = int(qmin + mini/s)
        # print("z: ", z)

        wq = (w/s + z).astype(np.int8)
        # Criando um número de 5 bits
        # print("wq: ", wq)
        # for e in wq:
        #     print("ola: ", e)
        #     bits = BitArray(int=int(e), length=b)
        #
        #     # Convertendo para um inteiro
        #     number = bits.int
        #     size = bits.length
        #
        #     print(number, size)  # Saída: 21
        print("or f: ", wq[0])
        return [wq, s, z]

    except Exception as e:
        print("min_max_quantization")
        print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, 0), type(e).__name__, e)

def quantization(layer, bits):

    try:
        if np.ndim(layer) == 1:
            return [layer, np.array([]), np.array([])]
        elif np.ndim(layer) == 2:
            r = min_max_quantization(layer, bits)
            return r
        elif np.ndim(layer) >= 3:
            u = []
            s = []
            z = []
            for i in range(len(layer)):
                r = quantization(layer[i], bits)
                u.append(r[0])
                s.append(r[1])
                z.append(r[2])
            return [np.array(u, dtype=np.int8), np.array(s, dtype=np.float16), np.array(z, dtype=np.float16)]

    except Exception as e:
        print("compression_methods")
        print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, 0), type(e).__name__, e)

def parameters_quantization_write(parameters, bits):

    try:

        quantized_parameters_list = []

        for parameter in parameters:
            quantized_parameters_list += quantization(parameter, bits)
        # print("quantizado: ", quantized_parameters_list[0])
        return quantized_parameters_list

    except Exception as e:
        print("parameters_quantization")
        print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, 0), type(e).__name__, e)

def inverse_parameter_quantization_reading(arrays, model_shape):
    try:
        # print("Recebidos: ", model_shape, arrays[0])
        sketched_paramters = []
        reconstructed_model = []
        parameter_index = 0
        sig_ind = 0
        j = 0
        for i in range(len(model_shape)):
            layer_shape = model_shape[i]
            print("i32: ", i*3+2)
            parameter = arrays[i*3]
            s = arrays[i*3 + 1]

            z = arrays[i*3 + 2]
            print("teste", parameter.shape, s.shape, z.shape, layer_shape, len(layer_shape))
            print("maior: ", i*3 + 2, len(arrays))
            parameter_layer = inverse_parameter_quantization(parameter, s, layer_shape, z)
            if parameter_layer is None:
                print("Pos ", i, i*3)
            reconstructed_model.append(parameter_layer)

        return reconstructed_model

    except Exception as e:
        print("inverse_parameter_quantization_reading")
        print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, 0), type(e).__name__, e)

def inverse_parameter_quantization(parameter, s, layer_index, z):
    try:
        if len(layer_index) == 1:
            print("u1")
            return parameter
        elif len(layer_index) == 2:
            print("u2")
            return min_max_dequantization(parameter, s, z)
        elif len(layer_index) == 3:
            print("u3")
            layers_l = []
            for i in range(len(parameter)):
                layers_l.append(min_max_dequantization(parameter[i], s[i], z[i]))
            return np.array(layers_l)
        elif len(layer_index) == 4:
            layers_l = []
            print("u4")
            for i in range(len(parameter)):
                layers_j = []
                for j in range(len(parameter[i])):
                    layers_j.append(min_max_dequantization(parameter[i][j], s[i][j], z[i][j]))
                layers_l.append(layers_j)
            return np.array(layers_l)

    except Exception as e:
        print("inverse_parameter_quantization")
        print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, 0), type(e).__name__, e)

def min_max_dequantization(wq, s, z):

    try:

        w_hat = s*(wq - z)

        return w_hat

    except Exception as e:
        print("min_max_dequantization")
        print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, 0), type(e).__name__, e)

# w = np.array([4, 1])
# quantize_linear_symmetric(w, 4)

# import torch
#
# # Definir um número float
# number = torch.tensor(0.456)
#
# # Realizar a quantização para um número com 4 bits
# quantized_number = torch.quantize_per_tensor(number, scale=0.1, zero_point=0, dtype=torch.qint8)
#
# print(quantized_number)  # Saída: tensor(0.4, dtype=torch.quint4x2)
# num_bytes = quantized_number.element_size() * quantized_number.numel()
#
# print(num_bytes)

# a, s, z = min_max_quantization(np.array([5,10]),4)
# b = min_max_dequantization(a, s, z)
# print(a, " s: ", s, " z: ", z)
# print("b: ", b)