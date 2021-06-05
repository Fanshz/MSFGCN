import numpy as np
from scipy.sparse.linalg import eigs
import scipy.sparse as sp

def scaled_laplacian(W):
    '''
    Normalized graph Laplacian function.
    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :return: np.matrix, [n_route, n_route].
    '''
    # d ->  diagonal degree matrix
    n, d = np.shape(W)[0], np.sum(W, axis=1)
    # L -> graph Laplacian
    L = -W
    L[np.diag_indices_from(L)] = d
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])
    # lambda_max \approx 2.0, the largest eigenvalues of L.
    # temp = eigs(L, k=1, which='LR')
    lambda_max = eigs(L, k=1, which='LR')[0][0].real##求出最大特征向量
    return np.mat(2 * L / lambda_max - np.identity(n))

def cheb_poly_approx(L, Ks, n):
    '''
    Chebyshev polynomials approximation function.
    :param L: np.matrix, [n_route, n_route], graph Laplacian.
    :param Ks: int, kernel size of spatial convolution.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, Ks*n_route].
    '''
    L0, L1 = np.mat(np.identity(n)), np.mat(np.copy(L))
    print(L0)
    print('gggggggggg')
    print(L1)
    L1 = L1 - np.diag(np.diag(L1))
    print("nbihao")
    print(L1)
    L1_ = L1 != 0
    L0_ = L0 != 0

    if Ks > 1:
        L_list = [np.copy(L0), np.copy(L1)]
        for i in range(Ks - 2):
            Ln = np.mat(2 * L * L1 - L0)
            print('hhhhhhhhhh-----', i+2)

            ##令Ln不为0的元素等于1
            Ln_ = Ln != 0
            Lnc = np.zeros_like(Ln)
            Lnc[Ln_] = 1

            ##令L1不为0的元素等于1
            L1c = np.zeros_like(L1)
            L1c[L1_] = 1

            ##令L0不为0的元素等于1
            L0c = np.zeros_like(L0)
            L0c[L0_] = 1

            ##Ln减去两步相邻矩阵
            Lnc = Lnc - L0c
            Lnc[Lnc < 0] = 0
            Lnc = Lnc - L1c##Ln减去两步相邻矩阵
            Lnc[Lnc < 0] = 0
            Lnc = Lnc.A
            Lnn = Ln.A * Lnc##对应位置相乘

            print(Lnn)
            L_list.append(np.copy(Lnn))
            L0, L1 = np.matrix(np.copy(L1)), np.matrix(np.copy(Ln))
            L0_, L1_ = np.matrix(np.copy(L1_)), np.matrix(np.copy(Ln_))
        # L_lsit [Ks, n*n], Lk [n, Ks*n]
        return np.concatenate(L_list, axis=-1)
    elif Ks == 1:
        return np.asarray(L0)
    else:
        raise ValueError(f'ERROR: the size of spatial kernel must be greater than 1, but received "{Ks}".')

# A = np.array([[0.0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
#                 [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
#                 [0, 1, 1, 0, 1, 0, 0, 1, 0, 0],
#                 [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
#                 [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
#                 [0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
#                 [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])

A = np.array([[0.0, 1, 0, 0, 0],
                [1, 0, 1, 1, 1],
                [0, 1, 0, 0, 0],
                [0, 1, 0, 0, 1],
                [0, 1, 0, 1, 0]])

L = scaled_laplacian(A)
Lk = cheb_poly_approx(A, 4, L.shape[0])
# Lk = cheb_poly_approx(A, 6, A.shape[0])