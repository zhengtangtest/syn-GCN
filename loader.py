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
        self.labels = [id2label[d[6]] for d in data] 
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
            pos = get_long_tensor(batch[1])
            ner = get_long_tensor(batch[2])
            deprel = get_long_tensor(batch[3])
            subj_positions = get_long_tensor(batch[4])
            obj_positions = get_long_tensor(batch[5])
            for i in range(len(words)):
                datalist += [Data(words=words[i], mask=torch.eq(words[i], 0), pos=pos[i], 
                    ner=ner[i], deprel=deprel[i], subj_position=subj_positions[i], 
                    obj_position=obj_positions[i], edge_index=torch.LongTensor(batch[7][i]),
                    rel=torch.LongTensor([batch[6][i]]))]

        self.data = DataLoader(datalist, batch_size=batch_size)

        print("{} batches created for {}".format(len(data), filename))

    def preprocess(self, data, vocab, opt):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in data:
            tokens = d['token']
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            tokens = ['$ROOT$'] + tokens
            # anonymize tokens
            ss, se = d['subj_start']+1, d['subj_end']+1
            os, oe = d['obj_start']+1, d['obj_end']+1
            tokens[ss:se+1] = ['SUBJ-'+d['subj_type']] * (se-ss+1)
            tokens[os:oe+1] = ['OBJ-'+d['obj_type']] * (oe-os+1)
            tokens = map_to_ids(tokens, vocab.word2id)
            pos = map_to_ids(['$ROOT$']+d['stanford_pos'], constant.POS_TO_ID)
            ner = map_to_ids(['$ROOT$']+d['stanford_ner'], constant.NER_TO_ID)
            deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            edge_index = [d['stanford_head'], list(range(1, len(d['stanford_head'])+1))]
            l = len(tokens)
            subj_positions = get_positions(ss, se, l)
            obj_positions = get_positions(os, oe, l)
            relation = constant.LABEL_TO_ID[d['relation']]
            processed += [(tokens, pos, ner, deprel, subj_positions, obj_positions, relation, edge_index)]
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

