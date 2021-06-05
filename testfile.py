# import tensorflow as tf
# # x = tf.constant(2)
# arr = tf.ones(3)
# arrb = tf.ones(3)
# sess = tf.Session()
# print(sess.run(tf.add_n([arr, arrb])))

# import tensorflow as tf
#
# x = tf.constant([1,2,3],dtype=tf.float32)
#
# with tf.Session() as sess:
#
#     print(sess.run(tf.nn.l2_loss(x)))

# import tensorflow as tf
#
# x = tf.Variable(1)
# y = tf.assign_add(x, 2)
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(y))
# print(sess.run(x))

# def fab(max):
#     n, a, b = 0, 0, 1
#     while n < max:
#         yield b  # 使用 yield
#         # print b
#         a, b = b, a + b
#         n = n + 1
#
#
# for n in fab(5):
#     print('hello')
#     print(n)
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
    lambda_max = eigs(L, k=1, which='LR')[0][0].real
    adj_mat = np.mat(2 * L / lambda_max - np.identity(n))
    adj_mat2 = adj_mat < 0
    print(adj_mat2)
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
    # L1__ = L1 > 0

    if Ks > 1:
        L_list = [np.copy(L0), np.copy(L1)]
        for i in range(Ks - 2):
            Ln = np.mat(2 * L * L1 - L0)
            print('hhhhhhhhhh-----', i+2)

            Ln_ = Ln != 0
            nonb = Ln_.nonzero()
            # print("nonbx", nonb[0])
            # print("nonby", nonb[1])
            Lnc = np.zeros_like(Ln)
            Lnc[Ln_] = 1

            nonb_ = L1_.nonzero()
            # print("nonbx", nonb_[0])
            # print("nonby", nonb_[1])
            L1c = np.zeros_like(L1)
            L1c[L1_] = 1

            nonb_ = L0_.nonzero()
            # print("nonbx", nonb_[0])
            # print("nonby", nonb_[1])
            L0c = np.zeros_like(L0)
            L0c[L0_] = 1

            Lnc = Lnc - L0c
            Lnc[Lnc < 0] = 0
            Lnc = Lnc - L1c
            Lnc[Lnc < 0] = 0
            Lnc = Lnc.A
            Lnn = Ln.A * Lnc

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

# A = np.array([[0.0,1,0,0,1,0],
#               [1,0,1,0,1,0],
#               [0,1,0,1,0,0],
#               [0,0,1,0,1,1],
#               [1,1,0,1,0,0],
#               [0,0,0,1,0,0]])
A = np.array([[0.0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
                [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
                [0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])

# D = A.sum(1)
# # d_inv_sqrt = np.power(D, -0.5).flatten()
# diags = np.diag(D)
# # normal =
# L = diags - A
# print(L)
# diags_ = np.power(diags, -0.5)
# diags_[np.isinf(diags_)] = 0.
# print(diags_)
# Lsys = diags_.dot(L).transpose().dot(diags_)
# print(Lsys)
# A=sp.csr_matrix(A)
L = scaled_laplacian(A)
Lk = cheb_poly_approx(L, 6, L.shape[0])
# print(Lk)