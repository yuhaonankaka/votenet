import bcolz
import numpy as np
import pickle
import os,json

from torch import nn
import torch
from torch.autograd import Variable


def read_target_vocab(filename):
    assert os.path.isfile(filename)
    with open(filename) as f:
        target_vocab = []
        data = json.load(f)
        num_objects = len(data)
        for i in range(num_objects):
            token = data[i]["token"]
            for t in token:
                if t not in target_vocab:
                    target_vocab.append(t)

    return target_vocab

glove_path = '/home/haonan/Downloads/glove.6B'
#
# words = []
# idx = 0
# word2idx = {}
# vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/glove.840B.dat', mode='w')
#
# with open(f'{glove_path}/glove.840B.300d.txt', 'rb') as f:
#     for l in f:
#         line = l.decode().split()
#         if len(line) == 301:
#             word = line[0]
#             words.append(word)
#             word2idx[word] = idx
#             idx += 1
#             vect = np.array(line[1:]).astype(np.float)
#             vectors.append(vect)
#
# vectors = bcolz.carray(vectors[1:].reshape((2195988, 300)), rootdir=f'{glove_path}/6B.300.dat', mode='w')
# vectors.flush()
# pickle.dump(words, open(f'{glove_path}/6B.300_words.pkl', 'wb'))
# pickle.dump(word2idx, open(f'{glove_path}/6B.300_idx.pkl', 'wb'))
scanreferpath = "/home/haonan/PycharmProjects/votenet_adl/scannet/meta_data_scanrefer/ScanRefer_filtered.json"


vectors = bcolz.open(f'{glove_path}/6B.300.dat')[:]
words = pickle.load(open(f'{glove_path}/6B.300_words.pkl', 'rb'))
word2idx = pickle.load(open(f'{glove_path}/6B.300_idx.pkl', 'rb'))

glove = {w: vectors[word2idx[w]] for w in words}

target_vocab = read_target_vocab(scanreferpath)

matrix_len = len(target_vocab)
weights_matrix = np.zeros((matrix_len+1, 300))
words_found = 0
target_vocab_word2indx = {}

for i, word in enumerate(target_vocab):
    target_vocab_word2indx[word] = i
    try:
        weights_matrix[i] = glove[word]
        words_found += 1
    except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6, size=(300, ))

print(words_found)

weights_matrix = torch.tensor(weights_matrix)

def getweights():
    return weights_matrix

def get_word2idx():
    return target_vocab_word2indx, matrix_len

def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


class LanguageNet(nn.Module):
    def __init__(self, weights_matrix, hidden_size, num_layers):
        super(LanguageNet, self).__init__()
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True)

    def forward(self, inp, hidden=None):
        inpt = self.embedding(inp)
        return self.gru(inpt, hidden)

    def init_hidden(self, batch_size):
        return torch.Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))




print("ok")