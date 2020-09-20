"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np

from utils import constant, helper, vocab

from torch_geometric.data import Data, DataLoader

class BatchLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab, evaluation=False):
        self.batch_size = batch_size
        self.vocab = vocab
        self.eval = evaluation
        self.opt = opt

        with open(filename) as infile:
            data = json.load(infile)
        data = self.preprocess(data, vocab, opt)
        data = sorted(data, key=lambda d: len(d[0]), reverse=True)
        
        id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])
        self.labels = [id2label[d[4]] for d in data] 
        self.num_examples = len(data)

        # chunk into batches
        data     = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.raw = data
        datalist = list()
        for i, batch in enumerate(data):
            batch = list(zip(*batch))
            assert len(batch) == 8
            
            # word dropout
            if not self.eval:
                words = [word_dropout(sent,  opt['word_dropout']) for sent in batch[0]]
            else:
                words = batch[0]
            # convert to tensors
            words = get_long_tensor(words)
            deprel = get_long_tensor(batch[1])
            subj_masks = get_long_tensor(batch[2])
            obj_masks = get_long_tensor(batch[3])
            rules = get_long_tensor(batch[6])
            golds = get_long_tensor(batch[7])
            for i in range(len(words)):
                datalist += [Data(words=words[i], mask=torch.eq(words[i], 0), 
                    deprel=deprel[i], d_mask=torch.eq(deprel[i], 0), 
                    subj_mask=torch.eq(subj_masks[i], 0), obj_mask=torch.eq(obj_masks[i], 0), 
                    edge_index=torch.LongTensor(batch[5][i]),
                    rel=torch.LongTensor([batch[4][i]]), rule=rules[i], gold=golds[i])]

        self.data = DataLoader(datalist, batch_size=batch_size)

        print("{} batches created for {}".format(len(data), filename))

    def preprocess(self, data, vocab, opt):
        # total = 0
        # pos = 0
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in data:
            if d[9]:
                # total += 1
                tokens = d[2]
                tokens = ['<ROOT>'] + tokens
                # anonymize tokens
                ss, se = d[3][0], d[3][-1]
                os, oe = d[4][0], d[4][-1]
                tokens[ss:se+1] = ['SUBJ'] * (se-ss+1)
                tokens[os:oe+1] = ['OBJ'] * (oe-os+1)
                tokens = map_to_ids(tokens, vocab.word2id)
                deprel = map_to_ids(d[8], constant.DEPREL_TO_ID)
                edge_index = d[7]
                temp = edge_index[0]
                edge_index[0] = edge_index[1]
                edge_index[1] = temp
                l = len(tokens)
                subj_positions = get_positions(ss, se, l)
                obj_positions = get_positions(os, oe, l)
                relation = 0 if d[1] == 'not_causal' else 1
                rule = map_to_ids(['<ROOT>'] + d[6], vocab.rule2id)
                gold = map_to_ids([d[9]], constant.GOLD_TO_ID)
                # if relation == 1:
                #     pos += 1
                processed += [(tokens, deprel, subj_positions, obj_positions, relation, edge_index, rule, gold)]
        # print (pos, total)
        return processed

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        #return 50
        return len(self.data)


def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))

def get_long_tensor(tokens_list):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = list()
    for i, s in enumerate(tokens_list):
        token = torch.LongTensor(token_len).fill_(constant.PAD_ID)
        token[:len(s)] = torch.LongTensor(s)
        tokens += [token]
    return tokens

def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]

