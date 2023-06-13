import numpy as np
from bitstring import BitArray
import sys
import flwr


def quantize_linear_symmetric(value, bits):
    qmin = -2**(bits - 1)
    qmax = 2**(bits - 1) - 1
    scale = max(abs(qmin), abs(qmax)) / max(abs(np.min(value)), abs(np.max(value)))
    value = np.round(value * scale)
    value = np.clip(value, qmin, qmax).astype(int)
    print("final: ", value)
    novo = []
    for e in value:
        e = int(e)
        print("Valor quantizado: ", e)

        bits_resulted = BitArray(int=e, length=bits)
        novo.append(bits_resulted)
        print("bits resulted: ", bits_resulted, type(bits_resulted), sys.getsizeof(bits_resulted))
        # Convertendo para um inteiro
        number = bits_resulted.int
        size = bits_resulted.length

        print("Covertido de volta: ", number, " tamanho: ", size)  # Saída: 21
    return bits_resulted

def quantize_linear_asymmetric(value, bits):
    qmin = -2**(bits - 1)
    qmax = 2**(bits - 1) - 1
    scale = (qmax - qmin) / (np.max(value) - np.min(value))
    zero_point = qmin - np.min(value) * scale
    value = np.round(value * scale + zero_point)
    value = np.clip(value, qmin, qmax)
    return value

def min_max_quantization(w, b):

    print("or: ", w)
    mini = np.minimum(w.min(), 0)
    maxim = np.maximum(w.max(), 0)
    w = np.clip(w, mini, maxim)
    print("de: ", w)
    s = (maxim - mini)/(2**b - 1)
    qmin = 0
    z = int(qmin + mini/s)
    print("z: ", z)

    wq = (w/s + z).astype(int)
    # Criando um número de 5 bits
    print("wq: ", wq)
    for e in wq:
        print("ola: ", e)
        bits = BitArray(int=int(e), length=b)

        # Convertendo para um inteiro
        number = bits.int
        size = bits.length

        print(number, size)  # Saída: 21
    return wq, s, z

def min_max_dequantization(wq, s, z):

    w_hat = s*(wq - z)

    return w_hat

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