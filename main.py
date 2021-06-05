# @Time     : Jan. 02, 2019 22:17
# @Author   : Veritas YIN
# @FileName : main.py
# @Version  : 1.0
# @Project  : Orion
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from os.path import join as pjoin

import tensorflow as tf

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
tf.Session(config=config)

from utils.math_graph import *
from utils.ms_math_graph import *
from data_loader.data_utils import *
from models.trainer import model_train
from models.tester import model_test

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_route', type=int, default=228)
parser.add_argument('--n_his', type=int, default=12)
parser.add_argument('--n_pred', type=int, default=9)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--epoch', type=int, default=51)
parser.add_argument('--save', type=int, default=50)
parser.add_argument('--ks', type=int, default=5)
parser.add_argument('--kt', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='RMSProp')
parser.add_argument('--graph', type=str, default='default')
parser.add_argument('--inf_mode', type=str, default='merge')

args = parser.parse_args()
print(f'Training configs: {args}')
save_result = 'D:/Project/MS-GCN_edit/MSGCN/output/result.txt'

n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
Ks, Kt = args.ks, args.kt
# blocks: settings of channel size in st_conv_blocks / bottleneck design
blocks = [[1, 32, 64], [64, 32, 128]]

# Load wighted adjacency matrix W
if args.graph == 'default':
    print(args.graph)
    # W = weight_matrix(pjoin('./dataset', f'PeMSD7_W_{n}.csv'))
    W = weight_matrix(pjoin('./dataset', f'W_{n}.csv'))#<class 'tuple'>: (228, 228)
else:
    # load customized graph weight matrix
    W = weight_matrix(pjoin('./dataset', args.graph))

# Calculate graph kernel
L = scaled_laplacian(W)
# Alternative approximation method: 1st approx - first_approx(W, n).
Lk = cheb_poly_approx(L, Ks, n)#<class 'tuple'>: (228, 684)
tf.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))##维护字典

# Calculate graph ms_kernel
ms_L = ms_scaled_laplacian(W)
# Alternative approximation method: 1st approx - first_approx(W, n).
ms_Lk = ms_cheb_poly_approx(ms_L, Ks, n)#<class 'tuple'>: (228, 684)
tf.add_to_collection(name='ms_graph_kernel', value=tf.cast(tf.constant(ms_Lk), tf.float32))##维护字典,添加改进的图卷积核

# Data Preprocessing
# data_file = f'PeMSD7_V_{n}.csv'
data_file = f'V_{n}.csv'
n_train, n_val, n_test = 34, 5, 5
PeMS = data_gen(pjoin('./dataset', data_file), (n_train, n_val, n_test), n, n_his + n_pred)##train：<class 'tuple'>: (9112, 21, 228, 1)
##val：<class 'tuple'>: (1340, 21, 228, 1)；test：<class 'tuple'>: (1340, 21, 228, 1)
print(f'>> Loading dataset with Mean: {PeMS.mean:.2f}, STD: {PeMS.std:.2f}')

if __name__ == '__main__':
    """Train Stage"""
    s_tra_time = time.time()
    model_train(PeMS, blocks, args)
    e_tra_time = time.time()

    """Test Stage"""
    # CUDA_VISIBLE_DEVICES = 0
    model_test(PeMS, PeMS.get_len('test'), n_his, n_pred, save_result, args.inf_mode)
    e_tes_time = time.time()
    # print(f'Training Time {e_tra_time - s_tra_time:.3f}s')
    print(f'Test Time {e_tes_time - e_tra_time:.3f}s')
