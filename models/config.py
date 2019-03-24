import os
from dataset import load_vocab
from utils import load_json, load_embeddings


class Config(object):
    def __init__(self, task):
        self.ckpt_path = './ckpt/{}/'.format(task)
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        source_dir = os.path.join('.', 'dataset', 'data', task)
        self.word_vocab, _ = load_vocab(os.path.join(source_dir, 'words.vocab'))
        self.char_vocab, _ = load_vocab(os.path.join(source_dir, 'chars.vocab'))
        self.vocab_size = len(self.word_vocab)        #单词表的大小
        self.char_vocab_size = len(self.char_vocab)   #字符的大小
        self.label_size = load_json(os.path.join(source_dir, 'label.json'))["label_size"]
        self.word_emb = load_embeddings(os.path.join(source_dir, 'glove.filtered.npz'))

    # log and model file paths
    max_to_keep = 5  # max model to keep while training
    no_imprv_patience = 100

    # word embeddings
    use_word_emb = True
    finetune_emb = False
    word_dim = 300

    # char embeddings
    use_char_emb = True
    char_dim = 50      #字符的维度是50
    char_rep_dim = 50

    # Convolutional neural networks filter size and height for char representation
    filter_sizes = [25, 25]  # sum of filter sizes should equal to char_out_size
    heights = [5, 5]  #这个不是很理解

    # highway network
    use_highway = False
    highway_num_layers = 2

    # model parameters
    num_layers = 10            #模型的层数
    num_units = 50             #模型的宽度
    num_units_last = 300       #模型最后一层的宽度

    # hyperparameters
    l2_reg = 0.001
    grad_clip = 5.0
    decay_lr = True
    lr = 0.01
    lr_decay = 0.05
    keep_prob = 1
