import json
import numpy as np


def load_json(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    except IOError:
        raise "ERROR: Unable to locate file {}".format(filename)


def load_embeddings(filename):
    '''

    :param filename: 文件路径名
    :return:
    '''
    try:
        with np.load(filename) as data:
            return data['embeddings']
    except IOError:
        raise 'ERROR: Unable to locate file {}.'.format(filename)




def batch_iter(dataset, batch_size):
    '''

    :param dataset:
    :param batch_size:
    :return:
    '''
    batch_x, batch_y = [], []
    for record in dataset:
        if len(batch_x) == batch_size:
            yield batch_x, batch_y
            batch_x, batch_y = [], []
        x = [tuple(value) for value in record["sentence"]]
        '''
        有助于理解
        >>> a=[123,[1,2,3]]
        >>> b=[456,[4,5,6]]
        >>> d=[a,b]
        >>> d
        [[123, [1, 2, 3]], [456, [4, 5, 6]]]
        >>> d=[tuple(a),tuple(b)]
        >>> d
        [(123, [1, 2, 3]), (456, [4, 5, 6])]
        >>> zip(*d)
        <zip object at 0x7f2cf6105ec8>
        >>> x=zip(*d)
        >>> for i,j in x:
        ...   print(i,j)
        ... 
        123 456
        [1, 2, 3] [4, 5, 6]
        >>> 
        '''
        x = zip(*x)
        y = record["label"]
        batch_x += [x]
        batch_y += [y]
    if len(batch_x) != 0:
        yield batch_x, batch_y


def _pad_sequences(sequences, pad_tok, max_length):
    '''
    此函数只是在第一维度进行paded   同样适应多维度的
    :param sequences: a generator of list or tuple
    :param pad_tok:  the char to pad with
    :param max_length:  最大的长度
    :return: 1. 一个二维的数组 每个元素拥有相同的长度
             2.每个元素实际的长度
    '''
    """Args:
        sequences: 
        pad_tok: 
    Returns:
       
    """
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq = list(seq)
        if len(seq) < max_length:
            seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        else:
            seq_ = seq[:max_length]
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]
    return sequence_padded, sequence_length


def pad_sequences(sequences, max_length, pad_tok, max_length_2=None, nlevels=1):
    '''

    :param sequences:  当nlevels=1：是一个二维数组  nlevels=2：是一个三维的数组
    :param max_length:
    :param pad_tok:
    :param max_length_2:
    :param nlevels:
    :return:
    '''
    """Args:
        sequences: a generator of list or tuple
        max_length: maximal length for a sentence allowed
        max_length_2: maximal length for a word allow, only for nLevels=2
        pad_tok: the char to pad with
        nlevels: depth of padding, 2 for the case where we have characters ids
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []
    if nlevels == 1:
        if max_length is None:
            #自动自动计算该batch 中的最大的长度
            max_length = max(map(lambda x: len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)
    elif nlevels == 2:
        if max_length_2 is None:
            max_length_2 = max([max(map(lambda x: len(x), seq)) for seq in sequences])  #这是获取所有单词的最大长度
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_2)
            sequence_padded += [sp]
            sequence_length += [sl]
        if max_length is None:
            max_length_sentence = max(map(lambda x: len(x), sequences))
        else:
            max_length_sentence = max_length
        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_2, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)
    return sequence_padded, sequence_length
