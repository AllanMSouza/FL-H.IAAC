# import numpy as np
# from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
#
# # Matriz densa
# dense_matrix = np.array([[1, 0, 0], [0, 0, 2], [3, 4, 0]], dtype=np.float64)
#
# # Esparsificação para CSR
# primeira = csr_matrix(dense_matrix, dtype=np.int8)
# segunda = coo_matrix(dense_matrix, dtype=np.int8)
# terceira = lil_matrix(dense_matrix, dtype=np.int8)
#
# # Tamanho em bytes antes e após a esparsificação
# dense_size = dense_matrix.nbytes
# sparse_size = primeira.data.nbytes + primeira.indices.nbytes + primeira.indptr.nbytes
# s2 = segunda.data.nbytes + segunda.row.nbytes + segunda.col.nbytes
# f = terceira.data.nbytes + terceira.rows.nbytes
# print("Tamanho antes da esparsificação:", dense_size, "bytes")
# print("Tamanho após a esparsificação:", sparse_size, "bytes")
# print(s2)
# print(f)
# print(np.array([12], dtype=np.int8).tobytes(), np.array([12], dtype=np.int8).nbytes)
#
#
# def exibir_bytes(numero):
#     # Convertendo o número inteiro para uma sequência de bytes
#     bytes_numero = numero.to_bytes((numero.bit_length() + 7) // 8, 'big')
#
#     # Exibindo os bytes
#     for byte in bytes_numero:
#         print(f'{byte:02X}', end=' ')  # Exibe cada byte em formato hexadecimal
#
#     print()  # Quebra de linha
#
# exibir_bytes(12)

import numpy as np

def gradient_sketching(gradients, compression_ratio):
    sketch = {}
    for layer, gradient in gradients.items():
        num_elements = gradient.size
        num_sketch_elements = int(num_elements * compression_ratio)
        sketch[layer] = np.random.choice(gradient, num_sketch_elements, replace=False)
    return sketch

# Exemplo de uso
# Suponha que você já tenha calculado os gradientes do seu modelo e armazenado em um dicionário chamado 'gradients'
# Suponha também que você deseje comprimir os gradientes para um fator de compressão de 0.5 (ou seja, reduzir pela metade)

# Chamada da função gradient_sketching
compression_ratio = 0.5
gradients = {'l1': np.array([1,2,4,5,6])}
compressed_gradients = gradient_sketching(gradients, compression_ratio)
print(compressed_gradients)

# Agora você pode usar os gradientes comprimidos para atualizar seu modelo
# ...


