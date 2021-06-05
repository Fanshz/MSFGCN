# @Time     : Jan. 12, 2019 17:45
# @Author   : Veritas YIN
# @FileName : layers.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

import tensorflow as tf
import numpy as np


def gconv(x, theta, Ks, c_in, c_out):
    '''
    Spectral-based graph convolution function.
    :param x: tensor, [batch_size, n_route, c_in]. (?, 228, 32)
    :param theta: tensor, [Ks*c_in, c_out], trainable kernel parameters.  (160, 32)
    :param Ks: int, kernel size of graph convolution.  5
    :param c_in: int, size of input channel.  32
    :param c_out: int, size of output channel.  32
    :return: tensor, [batch_size, n_route, c_out].
    '''
    # graph kernel: tensor, [n_route, Ks*n_route]
    kernel = tf.get_collection('ms_graph_kernel')[0]##(228, 1140)
    n = tf.shape(kernel)[0]
    # print(n)
    # x -> [batch_size, c_in, n_route] -> [batch_size*c_in, n_route]
    x_tmp = tf.reshape(tf.transpose(x, [0, 2, 1]), [-1, n])##1600(?, 228)
    # x_mul = x_tmp * ker -> [batch_size*c_in, Ks*n_route] -> [batch_size, c_in, Ks, n_route]
    x_mul = tf.reshape(tf.matmul(x_tmp, kernel), [-1, c_in, Ks, n])##矩阵乘
    # x_ker -> [batch_size, n_route, c_in, K_s] -> [batch_size*n_route, c_in*Ks]
    x_ker = tf.reshape(tf.transpose(x_mul, [0, 3, 1, 2]), [-1, c_in * Ks])
    # x_gconv -> [batch_size*n_route, c_out] -> [batch_size, n_route, c_out]
    x_gconv = tf.reshape(tf.matmul(x_ker, theta), [-1, n, c_out])#(?, 228, 32)
    return x_gconv

def gconv_att(x, ws, Ks, c_in, c_out):
    '''
    Spectral-based graph convolution function.
    :param x: tensor, [batch_size, n_route, c_in]. (?, 228, 32)
    :param theta: tensor, [Ks*c_in, c_out], trainable kernel parameters.  (160, 32)
    :param Ks: int, kernel size of graph convolution.  5
    :param c_in: int, size of input channel.  32
    :param c_out: int, size of output channel.  32
    :return: tensor, [batch_size, n_route, c_out].
    '''
    # graph kernel: tensor, [n_route, Ks*n_route]
    kernel = tf.get_collection('graph_kernel')[0]##(228, 1140)
    n = tf.shape(kernel)[0]
    # print(n)
    # x -> [batch_size, c_in, n_route] -> [batch_size*c_in, n_route]
    x_tmp = tf.reshape(tf.transpose(x, [0, 2, 1]), [-1, n])##1600(?, 228)
    # x_mul = x_tmp * ker -> [batch_size*c_in, Ks*n_route] -> [batch_size, c_in, Ks, n_route]
    x_mul = tf.reshape(tf.matmul(x_tmp, kernel), [-1, c_in, Ks, n])##矩阵乘
    # x_ker -> [batch_size, n_route, c_in, K_s] -> [batch_size*n_route, c_in*Ks]
    x_ker = tf.reshape(tf.transpose(x_mul, [0, 3, 1, 2]), [-1, c_in * Ks])
    # x_gconv -> [batch_size*n_route, c_out] -> [batch_size, n_route, c_out]
    # x_gconv = tf.reshape(tf.matmul(x_ker, theta), [-1, n, c_out])#(?, 228, 32)
    # return x_gconv
    #此处添加attention函数
    tmp = tf.matmul(x_ker, ws)
    x_gconv = tf.reshape(tmp, [-1, n, c_out])
    sum_all = tf.reduce_sum(x_gconv, keepdims=False)
    x_gcon = tf.divide(x_gconv, sum_all)
    return x_gcon

def ms_gconv(x, theta, Ks, c_in, c_out):
    '''
    Spectral-based graph convolution function.
    :param x: tensor, [batch_size, n_route, c_in]. (?, 228, 32)
    :param theta: tensor, [Ks*c_in, c_out], trainable kernel parameters.  (160, 32)
    :param Ks: int, kernel size of graph convolution.  5
    :param c_in: int, size of input channel.  32
    :param c_out: int, size of output channel.  32
    :return: tensor, [batch_size, n_route, c_out].
    '''
    # graph kernel: tensor, [n_route, Ks*n_route]
    # kernel = tf.get_collection('graph_kernel')[0]
    ms_kernel = tf.get_collection('ms_graph_kernel')[0]##(228, 1140)
    n = tf.shape(ms_kernel)[0]
    # print(n)
    # x -> [batch_size, c_in, n_route] -> [batch_size*c_in, n_route]
    x_tmp = tf.reshape(tf.transpose(x, [0, 2, 1]), [-1, n])##1600(?, 228)
    # x_mul = x_tmp * ker -> [batch_size*c_in, Ks*n_route] -> [batch_size, c_in, Ks, n_route]
    x_mul = tf.reshape(tf.matmul(x_tmp, ms_kernel), [-1, c_in, Ks, n])##矩阵乘
    # x_ker -> [batch_size, n_route, c_in, K_s] -> [batch_size*n_route, c_in*Ks]
    x_ker = tf.reshape(tf.transpose(x_mul, [0, 3, 1, 2]), [-1, c_in * Ks])
    # x_gconv -> [batch_size*n_route, c_out] -> [batch_size, n_route, c_out]
    x_gconv = tf.reshape(tf.matmul(x_ker, theta), [-1, n, c_out])#(?, 228, 32)
    return x_gconv

    # 此处添加attention机制
    # [batch_size*c_in, n_route]
    # mul_conv = tf.zeros_like(x_tmp)
    # for i in range(Ks):
    #     t_kernel = ms_kernel[:, i*n:(i+1)*n]
    #     mul_conv = tf.add(mul_conv, tf.matmul(x_tmp, t_kernel))
    # mul_conv = tf.expand_dims(tf.reshape(mul_conv, [-1, c_in, n]), 2)#(?, 228)->(?, 32, 1, 228)
    # mul_conv = tf.tile(mul_conv, (1, 1, Ks, 1))##(?, 32, 5, 228)
    # bi = tf.fill(tf.shape(mul_conv), 10.1)
    # attention_adj = tf.add(mul_conv, bi)
    # attention_adj = tf.nn.leaky_relu(attention_adj)
    # #此处添加标记变量
    # # attention_adj = tf.Variable(initial_value=tf.nn.leaky_relu(attention_adj), name='attention_adj', dtype=tf.float32)
    # # w_input = tf.get_variable('wt_input', shape=[1, 1, c_in, c_out], dtype=tf.float32)
    # x_ker_att = tf.divide(x_mul, attention_adj)
    # x_ker_att = tf.nn.relu(x_ker_att)
    # x_kern_att = tf.reshape(tf.transpose(x_ker_att, [0, 3, 1, 2]), [-1, c_in * Ks])
    # # x_kern_att = tf.transpose(x_ker_att, [0, 3, 1, 2])
    #
    # # x_ker -> [batch_size, n_route, c_in, K_s] -> [batch_size*n_route, c_in*Ks]
    # x_ker = tf.reshape(tf.transpose(x_mul, [0, 3, 1, 2]), [-1, c_in * Ks])
    # # x_ker = tf.transpose(x_mul, [0, 3, 1, 2])
    # ##合并到一起
    #
    # #[2 * batch_size * n_route, c_in * Ks]
    # x_ker = tf.concat([x_ker, x_kern_att], axis=1)
    #
    #
    # #theta  [Ks*c_in, c_out]->[2*Ks*c_in, c_out]
    # # theta = tf.tile(theta, (2, 1))
    # # x_gconv -> [batch_size*n_route, c_out] -> [batch_size, n_route, c_out]
    # x_gconv = tf.reshape(tf.matmul(x_ker, theta), [-1, n, c_out])#(?, 228, 32)
    # attention_adj = tf.transpose(attention_adj, [0, 3, ])
    # return x_gconv, attention_adj


def layer_norm(x, scope):
    '''
    Layer normalization function.
    :param x: tensor, [batch_size, time_step, n_route, channel].
    :param scope: str, variable scope.
    :return: tensor, [batch_size, time_step, n_route, channel].
    '''
    _, _, N, C = x.get_shape().as_list()
    mu, sigma = tf.nn.moments(x, axes=[2, 3], keep_dims=True)##计算x的均值和方差

    with tf.variable_scope(scope):
        gamma = tf.get_variable('gamma', initializer=tf.ones([1, 1, N, C]))
        beta = tf.get_variable('beta', initializer=tf.zeros([1, 1, N, C]))
        _x = (x - mu) / tf.sqrt(sigma + 1e-6) * gamma + beta
    return _x


def temporal_conv_layer(x, Kt, c_in, c_out, act_func='relu'):
    '''
    Temporal convolution layer.
    :param x: tensor, [batch_size, time_step, n_route, c_in].(?, 12, 228, 1)
    :param Kt: int, kernel size of temporal convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :param act_func: str, activation function.
    :return: tensor, [batch_size, time_step-Kt+1, n_route, c_out].
    '''
    __, T, n, _ = x.get_shape().as_list()##Tensor("strided_slice:0", shape=(?, 12, 228, 1), dtype=float32)

    if c_in > c_out:
        w_input = tf.get_variable('wt_input', shape=[1, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')
    elif c_in < c_out:#首先将input第3维扩大，用0扩充
        # if the size of input channel is less than the output,
        # padding x to the same size of output channel.
        # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=3)##(?, 12, 228, 1) -> (?, 12, 228, 32)
    else:
        x_input = x

    # keep the original input for residual connection.
    x_input = x_input[:, Kt - 1:T, :, :]##(?, 10, 228, 32)

    if act_func == 'GLU':
        # gated liner unit
        wt = tf.get_variable(name='wt', shape=[Kt, 1, c_in, 2 * c_out], dtype=tf.float32)##(3, 1, 1, 64)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
        bt = tf.get_variable(name='bt', initializer=tf.zeros([2 * c_out]), dtype=tf.float32)
        x_conv = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='VALID') + bt
        #x: [batch, in_height, in_width, in_channels],[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]
        #wt: [filter_height, filter_width, in_channels, out_channels][卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
        #strides: 卷积时在图像每一维的步长，这是一个一维的向量，长度4,[batch方向,height方向,width方向,channels方向
        #padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式 SAME 表示卷积后feature map与输入图像大小一致，VALID表示正常卷积，会按卷积核将feature map 边缘减小
        return (x_conv[:, :, :, 0:c_out] + x_input) * tf.nn.sigmoid(x_conv[:, :, :, -c_out:])
    else:
        wt = tf.get_variable(name='wt', shape=[Kt, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
        bt = tf.get_variable(name='bt', initializer=tf.zeros([c_out]), dtype=tf.float32)#(128,)
        x_conv = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='VALID') + bt
        if act_func == 'linear':
            return x_conv
        elif act_func == 'sigmoid':
            return tf.nn.sigmoid(x_conv)
        elif act_func == 'relu':
            return tf.nn.relu(x_conv + x_input)
        else:
            raise ValueError(f'ERROR: activation function "{act_func}" is not defined.')


def spatio_conv_layer(x, Ks, c_in, c_out):
    '''
    Spatial graph convolution layer.
    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param Ks: int, kernel size of spatial convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :return: tensor, [batch_size, time_step, n_route, c_out].
    '''
    _, T, n, _ = x.get_shape().as_list()

    if c_in > c_out:
        # bottleneck down-sampling
        w_input = tf.get_variable('ws_input', shape=[1, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')
    elif c_in < c_out:
        # if the size of input channel is less than the output,
        # padding x to the same size of output channel.
        # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=3)
    else:
        x_input = x

    ws = tf.get_variable(name='ws', shape=[Ks * c_in, c_out], dtype=tf.float32)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(ws))
    variable_summaries(ws, 'theta')
    bs = tf.get_variable(name='bs', initializer=tf.zeros([c_out]), dtype=tf.float32)
    # x -> [batch_size*time_step, n_route, c_in] -> [batch_size*time_step, n_route, c_out]
    x_gconv = gconv(tf.reshape(x, [-1, n, c_in]), ws, Ks, c_in, c_out) + bs
    # x_g -> [batch_size, time_step, n_route, c_out]
    x_gc = tf.reshape(x_gconv, [-1, T, n, c_out])
    # x_input = tf.tile(x_input, (2, 1, 1, 1))
    return tf.nn.relu(x_gc[:, :, :, 0:c_out] + x_input)

def ms_spatio_conv_layer(x, Ks, c_in, c_out):
    '''
    Spatial graph convolution layer.
    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param Ks: int, kernel size of spatial convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :return: tensor, [batch_size, time_step, n_route, c_out].
    '''
    _, T, n, _ = x.get_shape().as_list()

    if c_in > c_out:
        # bottleneck down-sampling
        w_input = tf.get_variable('ws_input', shape=[1, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')
    elif c_in < c_out:
        # if the size of input channel is less than the output,
        # padding x to the same size of output channel.
        # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=3)
    else:
        x_input = x

    ws = tf.get_variable(name='m_ws', shape=[Ks * c_in, c_out], dtype=tf.float32)
    tf.add_to_collection(name='m_weight_decay', value=tf.nn.l2_loss(ws))#将tensor对象放入同一个集合

    wa = tf.get_variable(name='m_wa', shape=[Ks * c_in, c_out], dtype=tf.float32)
    tf.add_to_collection(name='m_weight_attention', value=tf.nn.l2_loss(wa))

    variable_summaries(ws, 'm_theta')#做了一些画像
    variable_summaries(wa, 'm_theta')  # 做了一些画像

    bs = tf.get_variable(name='m_bs', initializer=tf.zeros([c_out]), dtype=tf.float32)
    b = tf.get_variable(name='m_b', initializer=tf.zeros([c_out]), dtype=tf.float32)
    # x -> [batch_size*time_step, n_route, c_in] -> [batch_size*time_step, n_route, c_out]
    # x_gconv = gconv(tf.reshape(x, [-1, n, c_in]), ws, Ks, c_in, c_out) + bs
    x_gcon_att = gconv_att(tf.reshape(x, [-1, n, c_in]), ws, Ks, c_in, c_out) + b
    # x_gcon = ms_gconv(tf.reshape(x, [-1, n, c_in]), wa, Ks, c_in, c_out) + bs
    x_gcon = gconv(tf.reshape(x, [-1, n, c_in]), wa, Ks, c_in, c_out) + bs
    x_gconv = tf.multiply(x_gcon, x_gcon_att)
    # x_gconv, attention_adj = gconv(tf.reshape(x, [-1, n, c_in]), ws, Ks, c_in, c_out) + bs
    # x_gconv = gconv(tf.reshape(x, [-1, n, c_in]), ws, Ks, c_in, c_out) + bs
    # x_g[2*batch_size, n_route, c_out] -> [batch_size, time_step, n_route, c_out]
    x_gc = tf.reshape(x_gconv, [-1, T, n, c_out])
    # x_input = tf.tile(x_input, (2, 1, 1, 1))
    return tf.nn.relu(x_gc[:, :, :, 0:c_out] + x_input)


def st_conv_block(x, Ks, Kt, channels, scope, keep_prob, act_func='GLU'):
    '''
    Spatio-temporal convolutional block, which contains two temporal gated convolution layers
    and one spatial graph convolution layer in the middle.
    :param x: tensor, batch_size, time_step, n_route, c_in].
    :param Ks: int, kernel size of spatial convolution.
    :param Kt: int, kernel size of temporal convolution.
    :param channels: list, channel configs of a single st_conv block.
    :param scope: str, variable scope.
    :param keep_prob: placeholder, prob of dropout.
    :param act_func: str, activation function.
    :return: tensor, [batch_size, time_step, n_route, c_out].
    '''
    c_si, c_t, c_oo = channels

    with tf.variable_scope(f'stn_block_{scope}_in'):
        x_s = temporal_conv_layer(x, Kt, c_si, c_t, act_func=act_func)#(?, 12, 228, 1)->(?, 10, 228, 32)
        # x_t = spatio_conv_layer(x_s, Ks, c_t, c_t)#(?, 10, 228, 32)
        x_t = ms_spatio_conv_layer(x_s, Ks, c_t, c_t)
        # x_t = spatio_conv_layer(x_s, Ks, c_t, c_t)
    with tf.variable_scope(f'stn_block_{scope}_out'):
        x_o = temporal_conv_layer(x_t, Kt, c_t, c_oo)#(?, 4, 228, 128)
        # x_o = ms_spatio_conv_layer(x_t, Ks, c_t, c_oo)
    x_ln = layer_norm(x_o, f'layer_norm_{scope}')
    return tf.nn.dropout(x_ln, keep_prob)


def fully_con_layer(x, n, channel, scope):
    '''
    Fully connected layer: maps multi-channels to one.
    :param x: tensor, [batch_size, 1, n_route, channel].
    :param n: int, number of route / size of graph.
    :param channel: channel size of input x.
    :param scope: str, variable scope.
    :return: tensor, [batch_size, 1, n_route, 1].
    '''
    w = tf.get_variable(name=f'w_{scope}', shape=[1, 1, channel, 1], dtype=tf.float32)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w))
    b = tf.get_variable(name=f'b_{scope}', initializer=tf.zeros([n, 1]), dtype=tf.float32)
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b


def output_layer(x, T, scope, act_func='GLU'):
    '''
    Output layer: temporal convolution layers attach with one fully connected layer,
    which map outputs of the last st_conv block to a single-step prediction.
    :param x: tensor, [batch_size, time_step, n_route, channel].
    :param T: int, kernel size of temporal convolution.
    :param scope: str, variable scope.
    :param act_func: str, activation function.
    :return: tensor, [batch_size, 1, n_route, 1].
    '''
    _, _, n, channel = x.get_shape().as_list()

    # maps multi-steps to one.
    with tf.variable_scope(f'{scope}_in'):
        x_i = temporal_conv_layer(x, T, channel, channel, act_func=act_func)#(?, 1, 228, 128)
    x_ln = layer_norm(x_i, f'layer_norm_{scope}')
    with tf.variable_scope(f'{scope}_out'):
        x_o = temporal_conv_layer(x_ln, 1, channel, channel, act_func='sigmoid')#(?, 1, 228, 128)
    # maps multi-channels to one.
    x_fc = fully_con_layer(x_o, n, channel, scope)
    return x_fc

"""
此处新建一个输出层函数
"""
# def output_layer2(x, T, scope, act_func='GLU'):
#     '''
#     Output layer: temporal convolution layers attach with one fully connected layer,
#     which map outputs of the last st_conv block to a single-step prediction.
#     :param x: tensor, [batch_size, time_step, n_route, channel].
#     :param T: int, kernel size of temporal convolution.
#     :param scope: str, variable scope.
#     :param act_func: str, activation function.
#     :return: tensor, [batch_size, 1, n_route, 1].
#     '''
#     _1, _2, n, channel = x.get_shape().as_list()
#
#     # maps multi-steps to one.
#     with tf.variable_scope(f'{scope}_in'):
#         x_i = temporal_conv_layer(x, T, channel, channel, act_func=act_func)
#
#     # weights = tf.get_variable('weights', shape=[1, 1, c_in, c_out], dtype=tf.float32)
#     # tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
#
#     x_ln = layer_norm(x_i, f'layer_norm_{scope}')
#     with tf.variable_scope(f'{scope}_out'):
#         x_o = temporal_conv_layer(x_ln, 1, channel, channel, act_func='sigmoid')
#     # maps multi-channels to one.
#     x_fc = fully_con_layer(x_o, n, channel, scope)
#     return x_fc

def variable_summaries(var, v_name):
    '''
    Attach summaries to a Tensor (for TensorBoard visualization).
    Ref: https://zhuanlan.zhihu.com/p/33178205
    :param var: tf.Variable().
    :param v_name: str, name of the variable.
    '''
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar(f'mean_{v_name}', mean)

        with tf.name_scope(f'stddev_{v_name}'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar(f'stddev_{v_name}', stddev)

        tf.summary.scalar(f'max_{v_name}', tf.reduce_max(var))
        tf.summary.scalar(f'min_{v_name}', tf.reduce_min(var))

        tf.summary.histogram(f'histogram_{v_name}', var)
