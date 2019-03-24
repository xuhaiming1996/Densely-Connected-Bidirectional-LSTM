import tensorflow as tf
from functools import reduce
from operator import mul
from tensorflow.contrib.rnn.python.ops.rnn_cell import _Linear as _linear
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from tensorflow.contrib.rnn.python.ops.rnn import stack_bidirectional_dynamic_rnn
from tensorflow.python.util import nest


class BiRNN:
    def __init__(self, num_units, scope='bi_rnn'):
        self.num_units = num_units
        self.cell_fw = LSTMCell(self.num_units)
        self.cell_bw = LSTMCell(self.num_units)
        self.scope = scope

    def __call__(self, inputs, seq_len, return_last_state=False):
        '''

        :param inputs:
        :param seq_len:
        :param return_last_state:
        :return:  最后返回不管是 outputs 还是state 都进行了拼接
        '''
        with tf.variable_scope(self.scope):
            if return_last_state:
                '''
                 (outputs, output_states)
                 其中：outputs:(output_fw, output_bw) 
                      output_states: (output_state_fw, output_state_bw)
                '''
                _, ((_, output_fw), (_, output_bw)) = bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, inputs,
                                                                                sequence_length=seq_len,
                                                                                dtype=tf.float32)
                output = tf.concat([output_fw, output_bw], axis=-1)
            else:
                (output_fw, output_bw), _ = bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, inputs,
                                                                      sequence_length=seq_len, dtype=tf.float32)
                output = tf.concat([output_fw, output_bw], axis=-1)
        return output


class StackedBiRNN:
    def __init__(self, num_layers, num_units, scope='stacked_bi_rnn'):
        self.num_layers = num_layers
        self.num_units = num_units
        self.cells_fw = [LSTMCell(self.num_units) for _ in range(self.num_layers)]
        self.cells_bw = [LSTMCell(self.num_units) for _ in range(self.num_layers)]
        self.scope = scope

    def __call__(self, inputs, seq_len):
        with tf.variable_scope(self.scope):
            output, *_ = stack_bidirectional_dynamic_rnn(self.cells_fw, self.cells_bw, inputs, sequence_length=seq_len,
                                                         dtype=tf.float32)
        return output


def dense(inputs, hidden_dim, use_bias=True, scope='dense'):
    '''

    :param inputs:
    :param hidden_dim:
    :param use_bias:
    :param scope:
    :return:
    '''
    with tf.variable_scope(scope):
        shape = tf.shape(inputs)
        dim = inputs.get_shape().as_list()[-1] #100
        # 64*label_size
        out_shape = [shape[idx] for idx in range(len(inputs.get_shape().as_list()) - 1)] + [hidden_dim]
        flat_inputs = tf.reshape(inputs, [-1, dim])

        w = tf.get_variable("weight", shape=[dim, hidden_dim], dtype=tf.float32)
        output = tf.matmul(flat_inputs, w)
        if use_bias:
            b = tf.get_variable("bias", shape=[hidden_dim], dtype=tf.float32, initializer=tf.constant_initializer(0.))
            output = tf.nn.bias_add(output, b)
        output = tf.reshape(output, out_shape)
        return output


def highway_layer(arg, bias, bias_start=0.0, scope=None, keep_prob=None, is_train=None):
    with tf.variable_scope(scope or "highway_layer"):
        d = arg.get_shape()[-1]
        trans = linear([arg], d, bias, bias_start=bias_start, scope='trans', keep_prob=keep_prob, is_train=is_train)
        trans = tf.nn.relu(trans)
        gate = linear([arg], d, bias, bias_start=bias_start, scope='gate', keep_prob=keep_prob, is_train=is_train)
        gate = tf.nn.sigmoid(gate)
        out = gate * trans + (1 - gate) * arg
        return out


def highway_network(arg, num_layers, bias, bias_start=0.0, scope=None, keep_prob=None, is_train=None):
    with tf.variable_scope(scope or "highway_network"):
        prev = arg
        cur = None
        for layer_idx in range(num_layers):
            cur = highway_layer(prev, bias, bias_start=bias_start, scope="layer_{}".format(layer_idx),
                                keep_prob=keep_prob, is_train=is_train)
            prev = cur
        return cur


def dropout(x, keep_prob, is_train, noise_shape=None, seed=None, name=None):
    '''

    :param x:
    :param keep_prob:
    :param is_train:
    :param noise_shape:
    :param seed:
    :param name:
    :return:
    '''
    with tf.name_scope(name or "dropout"):
        if keep_prob is not None and is_train is not None:
            out = tf.cond(is_train, lambda: tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed), lambda: x)   #这是一行好的代码
            return out
        return x

# 我这里感觉这个应该是 width
def conv1d(in_, filter_size, height, padding, is_train=None, keep_prob=None, scope=None):
    '''

    :param in_:  输入  这里猜测 in_的shape     [-1, max_len_sent,max_len_word,output size]
    :param filter_size: 感觉这里应该是卷积的数目
    :param height:
    :param padding:
    :param is_train:
    :param keep_prob:
    :param scope:
    :return:
    '''
    with tf.variable_scope(scope or "conv1d"):
        num_channels = in_.get_shape()[-1]
        filter_ = tf.get_variable("filter", shape=[1, height, num_channels, filter_size], dtype=tf.float32)
        bias = tf.get_variable("bias", shape=[filter_size], dtype=tf.float32)
        strides = [1, 1, 1, 1]
        print('许海明测试，dropout前 in_  shape: {}'.format(in_.get_shape().as_list()))
        if is_train is not None and keep_prob is not None:
            in_ = dropout(in_, keep_prob, is_train)

        print('许海明测试，dropout后 in_  shape: {}'.format(in_.get_shape().as_list()))

        # [batch, max_len_sent, max_len_word / filter_stride, char output size]
        '''
        in_      : [batch, in_height, in_width, in_channels]
        filter_  : [filter_height, filter_width, in_channels, out_channels]
        对这里的存在疑问
        '''
        xxc = tf.nn.conv2d(in_, filter_, strides, padding) + bias
        print('许海明测试，CNN 后的输出 xxc  shape: {}'.format(xxc.get_shape().as_list()))
        out = tf.reduce_max(tf.nn.relu(xxc), axis=2)  # max-pooling, [-1, max_len_sent, char output size]
        print('许海明测试，最后的返回结果 out!  shape: {}'.format(out.get_shape().as_list()))

        return out


def multi_conv1d(in_, filter_sizes, heights, padding, is_train=None, keep_prob=None, scope=None):
    '''
    :param in_:              输入的数据集
    :param filter_sizes:     过滤的大小
    :param heights:
    :param padding:
    :param is_train:
    :param keep_prob:
    :param scope:
    :return:
    '''

    '''
    filter_sizes = [25, 25]  # sum of filter sizes should equal to char_out_size
    heights = [5, 5]         #这个不是很理解 
    '''
    with tf.variable_scope(scope or "multi_conv1d"):
        assert len(filter_sizes) == len(heights)
        # 是一个列表
        outs = []
        for i, (filter_size, height) in enumerate(zip(filter_sizes, heights)):
            if filter_size == 0:
                continue
            out = conv1d(in_, filter_size, height, padding, is_train=is_train, keep_prob=keep_prob,
                         scope="conv1d_{}".format(i))
            outs.append(out)   # out的形状为 [-1, max_len_sent, char output size]


        concat_out = tf.concat(axis=2, values=outs)
        return concat_out


def linear(args, output_size, bias, bias_start=0.0, scope=None, squeeze=False, keep_prob=None, is_train=None):
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("args must be specified")
    if not nest.is_sequence(args):
        args = [args]
    flat_args = [flatten(arg, 1) for arg in args]
    if keep_prob is not None and is_train is not None:
        flat_args = [tf.cond(is_train, lambda: tf.nn.dropout(arg, keep_prob), lambda: arg) for arg in flat_args]
    with tf.variable_scope(scope or 'linear'):
        flat_out = _linear(flat_args, output_size, bias, bias_initializer=tf.constant_initializer(bias_start))
    out = reconstruct(flat_out, args[0], 1)
    if squeeze:
        out = tf.squeeze(out, [len(args[0].get_shape().as_list())-1])
    return out


def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat


def reconstruct(tensor, ref, keep):
    ref_shape = ref.get_shape().as_list()
    tensor_shape = tensor.get_shape().as_list()
    ref_stop = len(ref_shape) - keep
    tensor_start = len(tensor_shape) - keep
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out
